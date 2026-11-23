"""Leakage-resistant copy-number inputs and paired DepPLM sensitivity runs.

The primary corrected campaign intentionally remains unchanged.  This module
provides the separately labelled CNV sensitivity layer: it aligns the pinned
DepMap 24Q4 absolute gene-copy-number table to an explicit cell/gene cohort,
removes cells that have no source row, and fits imputation, scaling, and PCA on
training rows only.  Its command-line runner compares baseline and baseline+CNV
on identical cells and splits and writes hash-bound outer-test predictions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


CNV_FILENAME = "OmicsAbsoluteCNGene.csv"
CNV_SIZE_BYTES = 238_808_692
CNV_SHA256 = "5f530c797bf12bce68cccc3e763b29491c368bbbced8971733bed807a13b8c13"
CNV_SOURCE_URL = "https://ndownloader.figshare.com/files/51065303"
CNV_RELEASE_DOI = "10.25452/figshare.plus.27993248.v1"

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DEFAULT_CNV_PATH = ROOT / "data" / "depmap" / CNV_FILENAME
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "cnv_sensitivity_v1"
CANONICAL_SEEDS = (42, 43, 44)
CANONICAL_FOLDS = 5
CANONICAL_CNV_COMPONENTS = 32
CANONICAL_EXPR_COMPONENTS = 128
CANONICAL_INNER_EPOCHS = 20
CANONICAL_PATIENCE = 5
CANONICAL_BATCH_SIZE = 64
CANONICAL_GENE_SUBSAMPLE = 6_000
CANONICAL_TOP_MUT = 5_000
CANONICAL_GRID = (
    {"hidden": 128, "lr": 1e-3, "wd": 1e-5, "lamb_sel": 0.1},
    {"hidden": 128, "lr": 3e-4, "wd": 1e-5, "lamb_sel": 0.1},
    {"hidden": 256, "lr": 1e-3, "wd": 1e-5, "lamb_sel": 0.1},
    {"hidden": 256, "lr": 3e-4, "wd": 1e-5, "lamb_sel": 0.1},
)
PRIMARY_METRICS = (
    "mse",
    "mae",
    "residual_r2_vs_train_gene_mean",
    "residual_pearson_pooled",
    "residual_cellwise_pearson_mean",
    "cellwise_pearson_mean",
    "cellwise_spearman_mean",
)

_GENE_COLUMN = re.compile(r"^(?P<symbol>.+) \((?P<entrez>[0-9]+)\)$")


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(int(chunk_size)):
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


def _unique_strings(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result:
        raise ValueError(f"{label} must not be empty")
    if any(not value for value in result):
        raise ValueError(f"{label} must not contain empty identifiers")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must contain unique identifiers")
    return result


def _unique_indices(
    values: Sequence[int] | np.ndarray,
    n_rows: int,
    label: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{label} must be a non-empty one-dimensional array")
    if np.unique(result).size != result.size:
        raise ValueError(f"{label} must not contain duplicates")
    if int(result.min()) < 0 or int(result.max()) >= int(n_rows):
        raise IndexError(f"{label} contains an index outside 0..{n_rows - 1}")
    return result


@dataclass(frozen=True)
class CnvHeader:
    id_column: str
    raw_gene_columns: tuple[str, ...]
    gene_symbols: tuple[str, ...]
    entrez_ids: tuple[str, ...]


def read_cnv_header(path: str | Path) -> CnvHeader:
    """Read and validate the DepMap ``SYMBOL (ENTREZ)`` header."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError("CNV source is empty") from exc
    if len(header) < 2:
        raise ValueError("CNV source must contain an ID column and gene columns")
    if len(set(header)) != len(header):
        raise ValueError("CNV source contains duplicate raw column names")

    symbols: list[str] = []
    entrez: list[str] = []
    for column in header[1:]:
        match = _GENE_COLUMN.fullmatch(column)
        if match is None:
            raise ValueError(
                f"CNV gene column does not match 'SYMBOL (ENTREZ)': {column!r}"
            )
        symbols.append(match.group("symbol"))
        entrez.append(match.group("entrez"))
    if len(set(symbols)) != len(symbols):
        raise ValueError("CNV source contains duplicate gene symbols")
    if len(set(entrez)) != len(entrez):
        raise ValueError("CNV source contains duplicate Entrez identifiers")
    return CnvHeader(
        id_column=header[0],
        raw_gene_columns=tuple(header[1:]),
        gene_symbols=tuple(symbols),
        entrez_ids=tuple(entrez),
    )


def _validate_source(
    path: Path,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
) -> dict[str, str | int]:
    size = path.stat().st_size
    if expected_size is not None and size != int(expected_size):
        raise RuntimeError(f"{path} has {size} bytes; expected {expected_size}")
    digest = _sha256_file(path)
    if expected_sha256 is not None and digest != str(expected_sha256):
        raise RuntimeError(f"{path} has SHA-256 {digest}; expected {expected_sha256}")
    return {"path": str(path), "bytes": int(size), "sha256": digest}


def _read_source_cell_ids(path: Path) -> tuple[str, ...]:
    # The pinned file's first header field is empty (pandas exposes it as
    # ``Unnamed: 0``), so bind the ModelID column by its validated position.
    frame = pd.read_csv(path, usecols=[0], dtype=str)
    identifiers = frame.iloc[:, 0]
    if identifiers.isna().any():
        raise ValueError("CNV source contains a missing ModelID")
    values = tuple(identifiers.astype(str))
    if any(not value for value in values):
        raise ValueError("CNV source contains an empty ModelID")
    if len(set(values)) != len(values):
        raise ValueError("CNV source contains duplicate ModelID rows")
    return values


@dataclass(frozen=True)
class CnvCoverage:
    requested_cell_ids: tuple[str, ...]
    requested_genes: tuple[str, ...]
    matched_cell_ids: tuple[str, ...]
    matched_genes: tuple[str, ...]
    missing_cell_ids: tuple[str, ...]
    missing_genes: tuple[str, ...]
    request_cell_indices: np.ndarray
    request_gene_indices: np.ndarray
    source_row_indices: np.ndarray
    source_column_indices: np.ndarray
    source_cell_count: int
    source_gene_count: int

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "DepPLM-CNV-coverage-v1",
            "requested_cells": len(self.requested_cell_ids),
            "matched_cells": len(self.matched_cell_ids),
            "missing_cells": len(self.missing_cell_ids),
            "cell_coverage_fraction": len(self.matched_cell_ids)
            / len(self.requested_cell_ids),
            "requested_genes": len(self.requested_genes),
            "matched_genes": len(self.matched_genes),
            "missing_genes": len(self.missing_genes),
            "gene_coverage_fraction": len(self.matched_genes)
            / len(self.requested_genes),
            "source_cells": int(self.source_cell_count),
            "source_genes": int(self.source_gene_count),
            "requested_cell_ids_sha256": _hash_strings(self.requested_cell_ids),
            "matched_cell_ids_sha256": _hash_strings(self.matched_cell_ids),
            "requested_genes_sha256": _hash_strings(self.requested_genes),
            "matched_genes_sha256": _hash_strings(self.matched_genes),
            "request_cell_indices_sha256": _hash_array(
                self.request_cell_indices.astype("<i8", copy=False)
            ),
            "request_gene_indices_sha256": _hash_array(
                self.request_gene_indices.astype("<i8", copy=False)
            ),
            "source_row_indices_sha256": _hash_array(
                self.source_row_indices.astype("<i8", copy=False)
            ),
            "source_column_indices_sha256": _hash_array(
                self.source_column_indices.astype("<i8", copy=False)
            ),
        }


def inspect_cnv_coverage(
    path: str | Path,
    requested_cell_ids: Sequence[str],
    requested_genes: Sequence[str],
) -> CnvCoverage:
    """Inspect row/column overlap without loading the numeric CNV matrix."""
    source = Path(path)
    cells = _unique_strings(requested_cell_ids, "requested_cell_ids")
    genes = _unique_strings(requested_genes, "requested_genes")
    header = read_cnv_header(source)
    source_cells = _read_source_cell_ids(source)
    cell_to_row = {cell_id: row for row, cell_id in enumerate(source_cells)}
    gene_to_column = {
        symbol: column for column, symbol in enumerate(header.gene_symbols, start=1)
    }

    request_cell_indices = np.asarray(
        [index for index, cell_id in enumerate(cells) if cell_id in cell_to_row],
        dtype=np.int64,
    )
    request_gene_indices = np.asarray(
        [index for index, gene in enumerate(genes) if gene in gene_to_column],
        dtype=np.int64,
    )
    matched_cells = tuple(cells[int(index)] for index in request_cell_indices)
    matched_genes = tuple(genes[int(index)] for index in request_gene_indices)
    return CnvCoverage(
        requested_cell_ids=cells,
        requested_genes=genes,
        matched_cell_ids=matched_cells,
        matched_genes=matched_genes,
        missing_cell_ids=tuple(cell for cell in cells if cell not in cell_to_row),
        missing_genes=tuple(gene for gene in genes if gene not in gene_to_column),
        request_cell_indices=request_cell_indices,
        request_gene_indices=request_gene_indices,
        source_row_indices=np.asarray(
            [cell_to_row[cell] for cell in matched_cells], dtype=np.int64
        ),
        source_column_indices=np.asarray(
            [gene_to_column[gene] for gene in matched_genes], dtype=np.int64
        ),
        source_cell_count=len(source_cells),
        source_gene_count=len(header.gene_symbols),
    )


@dataclass(frozen=True)
class CnvCohort:
    values: np.ndarray
    coverage: CnvCoverage
    source: Mapping[str, str | int]

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self.coverage.matched_cell_ids

    @property
    def gene_names(self) -> tuple[str, ...]:
        return self.coverage.matched_genes

    def manifest(self) -> dict[str, Any]:
        finite = np.isfinite(self.values)
        return {
            "schema": "DepPLM-CNV-cohort-v1",
            "source": dict(self.source),
            "coverage": self.coverage.manifest(),
            "shape": list(self.values.shape),
            "dtype": str(self.values.dtype),
            "values_sha256": _hash_array(self.values),
            "finite_values": int(finite.sum()),
            "missing_values": int((~finite).sum()),
            "fraction_missing": float((~finite).mean()),
        }


def load_cnv_cohort(
    path: str | Path,
    requested_cell_ids: Sequence[str],
    requested_genes: Sequence[str],
    *,
    expected_size: int | None = CNV_SIZE_BYTES,
    expected_sha256: str | None = CNV_SHA256,
) -> CnvCohort:
    """Load CNV values in requested order, excluding absent source rows.

    Cells with no source row are returned in ``coverage.missing_cell_ids`` and
    are not imputed.  Individual missing gene values are retained as NaN for
    training-only imputation by :func:`fit_cnv_pca`.
    """
    source = Path(path)
    source_record = _validate_source(
        source,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    )
    coverage = inspect_cnv_coverage(source, requested_cell_ids, requested_genes)
    if not coverage.matched_cell_ids:
        raise ValueError("none of the requested cells has a CNV source row")
    if not coverage.matched_genes:
        raise ValueError("none of the requested genes has a CNV source column")

    header = read_cnv_header(source)
    raw_by_symbol = dict(zip(header.gene_symbols, header.raw_gene_columns))
    selected_columns = [raw_by_symbol[gene] for gene in coverage.matched_genes]
    selected_positions = coverage.source_column_indices.tolist()
    dtype = {column: np.float32 for column in selected_columns}
    frame = pd.read_csv(
        source,
        usecols=[0, *selected_positions],
        dtype=dtype,
    )
    identifiers = frame.iloc[:, 0]
    if identifiers.isna().any():
        raise ValueError("CNV source contains a missing ModelID")
    if not identifiers.is_unique:
        raise ValueError("CNV source contains duplicate ModelID rows")
    frame.index = identifiers.astype(str)
    frame = frame.iloc[:, 1:]
    frame = frame.reindex(index=coverage.matched_cell_ids, columns=selected_columns)
    if frame.index.to_list() != list(coverage.matched_cell_ids):
        raise RuntimeError("CNV row alignment failed")
    values = frame.to_numpy(dtype=np.float32, copy=True)
    if values.shape != (
        len(coverage.matched_cell_ids),
        len(coverage.matched_genes),
    ):
        raise RuntimeError("CNV matrix shape does not match the alignment manifest")
    return CnvCohort(values=values, coverage=coverage, source=source_record)


@dataclass(frozen=True)
class FittedCnvPCA:
    mean_: np.ndarray
    scale_: np.ndarray
    pca_mean_: np.ndarray
    components_: np.ndarray
    explained_variance_ratio_: np.ndarray
    retained_feature_indices: np.ndarray
    dropped_all_missing_feature_indices: np.ndarray
    n_input_features: int
    n_output_components: int
    fit_train_indices_sha256: str
    fit_training_values_sha256: str
    fit_training_shape: tuple[int, int]
    fit_seed: int
    pca_solver: str
    sklearn_version: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_input_features:
            raise ValueError("CNV matrix has an incompatible feature dimension")
        retained = matrix[:, self.retained_feature_indices]
        clean = np.where(np.isfinite(retained), retained, self.mean_)
        standardized = (clean - self.mean_) / self.scale_
        scores = (standardized - self.pca_mean_) @ self.components_.T
        if scores.shape[1] < self.n_output_components:
            scores = np.pad(
                scores,
                ((0, 0), (0, self.n_output_components - scores.shape[1])),
                mode="constant",
            )
        return scores.astype(np.float32, copy=False)


def fit_cnv_pca(
    values: np.ndarray,
    train_indices: Sequence[int] | np.ndarray,
    *,
    n_components: int = 32,
    seed: int = 42,
) -> FittedCnvPCA:
    """Fit CNV imputation, scaling, and PCA using training rows only."""
    from sklearn import __version__ as sklearn_version
    from sklearn.decomposition import PCA

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("values must be a non-empty two-dimensional CNV matrix")
    train = _unique_indices(train_indices, matrix.shape[0], "train_indices")
    if train.size < 2:
        raise ValueError("at least two training cells are required for CNV PCA")
    if int(n_components) < 1:
        raise ValueError("n_components must be positive")

    training_values = matrix[train]
    finite_counts = np.isfinite(training_values).sum(axis=0)
    retained_indices = np.flatnonzero(finite_counts > 0).astype(np.int64)
    dropped_indices = np.flatnonzero(finite_counts == 0).astype(np.int64)
    if retained_indices.size == 0:
        raise ValueError("all CNV features are missing in the training rows")
    retained_train = training_values[:, retained_indices]
    finite = np.isfinite(retained_train)
    counts = finite.sum(axis=0)
    totals = np.where(finite, retained_train, 0.0).sum(axis=0, dtype=np.float64)
    mean = totals / counts
    centered = np.where(finite, retained_train - mean, 0.0)
    scale = np.sqrt(np.square(centered).sum(axis=0, dtype=np.float64) / counts)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    clean = np.where(finite, retained_train, mean)
    standardized = (clean - mean) / scale
    fitted_components = min(
        int(n_components), standardized.shape[0], standardized.shape[1]
    )
    pca = PCA(
        n_components=fitted_components,
        svd_solver="full",
        random_state=int(seed),
    )
    pca.fit(standardized)
    return FittedCnvPCA(
        mean_=mean.astype(np.float64, copy=False),
        scale_=scale.astype(np.float64, copy=False),
        pca_mean_=pca.mean_.astype(np.float64, copy=False),
        components_=pca.components_.astype(np.float64, copy=False),
        explained_variance_ratio_=pca.explained_variance_ratio_.astype(
            np.float64, copy=False
        ),
        retained_feature_indices=retained_indices,
        dropped_all_missing_feature_indices=dropped_indices,
        n_input_features=int(matrix.shape[1]),
        n_output_components=int(n_components),
        fit_train_indices_sha256=_hash_array(train.astype("<i8", copy=False)),
        fit_training_values_sha256=_hash_array(training_values),
        fit_training_shape=tuple(int(value) for value in training_values.shape),
        fit_seed=int(seed),
        pca_solver="full",
        sklearn_version=str(sklearn_version),
    )


def cnv_fit_manifest(
    fitted: FittedCnvPCA,
    gene_names: Sequence[str],
    train_indices: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Return a hash-bound record of one training-fold CNV transform."""
    genes = _unique_strings(gene_names, "gene_names")
    if len(genes) != fitted.n_input_features:
        raise ValueError("gene_names do not match the fitted CNV feature count")
    train = _unique_indices(train_indices, 2**63 - 1, "train_indices")
    train_indices_sha256 = _hash_array(train.astype("<i8", copy=False))
    if train_indices_sha256 != fitted.fit_train_indices_sha256:
        raise ValueError("train_indices do not match the rows used to fit CNV PCA")
    retained_names = tuple(genes[int(i)] for i in fitted.retained_feature_indices)
    dropped_names = tuple(
        genes[int(i)] for i in fitted.dropped_all_missing_feature_indices
    )
    return {
        "schema": "DepPLM-CNV-training-transform-v1",
        "fit_scope": "training_rows_only",
        "train_indices_sha256": train_indices_sha256,
        "training_values_sha256": fitted.fit_training_values_sha256,
        "training_values_shape": list(fitted.fit_training_shape),
        "n_train_rows": int(train.size),
        "n_input_features": int(fitted.n_input_features),
        "n_retained_features": int(fitted.retained_feature_indices.size),
        "n_dropped_all_missing_features": int(
            fitted.dropped_all_missing_feature_indices.size
        ),
        "n_output_components": int(fitted.n_output_components),
        "n_fitted_components": int(fitted.components_.shape[0]),
        "seed": int(fitted.fit_seed),
        "pca_solver": fitted.pca_solver,
        "sklearn_version": fitted.sklearn_version,
        "gene_names_sha256": _hash_strings(genes),
        "retained_gene_names_sha256": _hash_strings(retained_names),
        "dropped_gene_names_sha256": _hash_strings(dropped_names),
        "retained_feature_indices_sha256": _hash_array(
            fitted.retained_feature_indices.astype("<i8", copy=False)
        ),
        "dropped_feature_indices_sha256": _hash_array(
            fitted.dropped_all_missing_feature_indices.astype("<i8", copy=False)
        ),
        "mean_sha256": _hash_array(fitted.mean_),
        "scale_sha256": _hash_array(fitted.scale_),
        "pca_mean_sha256": _hash_array(fitted.pca_mean_),
        "components_sha256": _hash_array(fitted.components_),
        "explained_variance_ratio_sha256": _hash_array(
            fitted.explained_variance_ratio_
        ),
    }


@dataclass(frozen=True)
class PairedCnvCohort:
    cell_ids: tuple[str, ...]
    original_cell_indices: np.ndarray
    baseline_values: np.ndarray
    cnv_values: np.ndarray
    cnv_gene_names: tuple[str, ...]


def make_paired_cnv_cohort(
    requested_cell_ids: Sequence[str],
    baseline_cell_values: np.ndarray,
    cnv: CnvCohort,
) -> PairedCnvCohort:
    """Restrict the baseline arm to exactly the cells present in the CNV arm."""
    requested = _unique_strings(requested_cell_ids, "requested_cell_ids")
    baseline = np.asarray(baseline_cell_values)
    if baseline.ndim < 1 or baseline.shape[0] != len(requested):
        raise ValueError(
            "baseline_cell_values first dimension must match requested cells"
        )
    if cnv.coverage.requested_cell_ids != requested:
        raise ValueError(
            "CNV coverage was constructed for a different cell cohort/order"
        )
    indices = cnv.coverage.request_cell_indices
    paired_baseline = np.ascontiguousarray(baseline[indices])
    paired_cnv = np.ascontiguousarray(cnv.values)
    if paired_baseline.shape[0] != paired_cnv.shape[0]:
        raise RuntimeError("baseline and CNV arm row counts differ after pairing")
    return PairedCnvCohort(
        cell_ids=cnv.cell_ids,
        original_cell_indices=indices.copy(),
        baseline_values=paired_baseline,
        cnv_values=paired_cnv,
        cnv_gene_names=cnv.gene_names,
    )


def paired_split_manifest(
    paired: PairedCnvCohort,
    train_indices: Sequence[int] | np.ndarray,
    test_indices: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Validate one shared split and bind both arms to the same cell identities."""
    n_cells = len(paired.cell_ids)
    train = _unique_indices(train_indices, n_cells, "train_indices")
    test = _unique_indices(test_indices, n_cells, "test_indices")
    if set(train).intersection(test):
        raise ValueError("paired train and test indices overlap")
    if set(np.concatenate([train, test])) != set(range(n_cells)):
        raise ValueError("paired train/test indices must partition the paired cohort")
    if (
        paired.baseline_values.shape[0] != n_cells
        or paired.cnv_values.shape[0] != n_cells
    ):
        raise ValueError("paired arm matrices do not match the cell identity vector")
    train_cells = tuple(paired.cell_ids[int(index)] for index in train)
    test_cells = tuple(paired.cell_ids[int(index)] for index in test)
    return {
        "schema": "DepPLM-CNV-paired-split-v1",
        "pairing_policy": "identical_cells_and_split_indices_for_both_arms",
        "n_cells": int(n_cells),
        "n_train": int(train.size),
        "n_test": int(test.size),
        "cell_ids_sha256": _hash_strings(paired.cell_ids),
        "original_cell_indices_sha256": _hash_array(
            paired.original_cell_indices.astype("<i8", copy=False)
        ),
        "train_indices_sha256": _hash_array(train.astype("<i8", copy=False)),
        "test_indices_sha256": _hash_array(test.astype("<i8", copy=False)),
        "train_cell_ids_sha256": _hash_strings(train_cells),
        "test_cell_ids_sha256": _hash_strings(test_cells),
        "baseline_cell_ids_sha256": _hash_strings(paired.cell_ids),
        "cnv_cell_ids_sha256": _hash_strings(paired.cell_ids),
        "baseline_cnv_cell_ids_identical": True,
    }


def restrict_base_to_cnv_cohort(base: Any, cnv: CnvCohort) -> Any:
    """Return a corrected-runner ``BaseData`` restricted to CNV-covered cells.

    The function imports the corrected runner lazily so the alignment and PCA
    primitives remain usable in lightweight environments without PyTorch.
    """
    from run_corrected_experiments import BaseData

    if tuple(str(value) for value in base.cell_ids) != cnv.coverage.requested_cell_ids:
        raise ValueError(
            "base cell identities/order differ from the CNV request cohort"
        )
    indices = cnv.coverage.request_cell_indices.astype(np.int64, copy=False)
    restricted_ids = [str(base.cell_ids[int(index)]) for index in indices]
    if tuple(restricted_ids) != cnv.cell_ids:
        raise RuntimeError(
            "CNV and baseline cell identities diverged during restriction"
        )
    return BaseData(
        gene_effect=np.ascontiguousarray(
            base.gene_effect[:, indices], dtype=np.float32
        ),
        expression_raw=np.ascontiguousarray(base.expression_raw[indices]),
        mutations_by_cell=[base.mutations_by_cell[int(index)] for index in indices],
        embeddings=np.ascontiguousarray(base.embeddings),
        edge_index=np.ascontiguousarray(base.edge_index),
        genes=[str(value) for value in base.genes],
        cell_ids=restricted_ids,
        lineages=np.ascontiguousarray(base.lineages[indices]),
        source_cell_indices=np.ascontiguousarray(
            base.source_cell_indices[indices], dtype=np.int64
        ),
        source_gene_indices=np.ascontiguousarray(
            base.source_gene_indices, dtype=np.int64
        ),
        expression_gene_names=[str(value) for value in base.expression_gene_names],
    )


def augment_fold_data_with_cnv(
    fold_data: Any,
    cnv_values: np.ndarray,
    train_indices: Sequence[int] | np.ndarray,
    *,
    n_components: int = CANONICAL_CNV_COMPONENTS,
    seed: int = 42,
) -> tuple[Any, FittedCnvPCA]:
    """Append training-only CNV PCs to an already fitted expression branch."""
    expression = np.asarray(fold_data.expression, dtype=np.float32)
    matrix = np.asarray(cnv_values)
    if expression.ndim != 2 or expression.shape[0] != matrix.shape[0]:
        raise ValueError("fold expression rows and CNV rows must be identical")
    fitted = fit_cnv_pca(
        matrix,
        train_indices,
        n_components=int(n_components),
        seed=int(seed),
    )
    scores = fitted.transform(matrix)
    augmented = np.ascontiguousarray(
        np.concatenate([expression, scores], axis=1), dtype=np.float32
    )
    return replace(fold_data, expression=augmented), fitted


def paired_metric_deltas(
    baseline_metrics: Mapping[str, Any],
    cnv_metrics: Mapping[str, Any],
) -> dict[str, float]:
    """Return paired CNV-minus-baseline deltas with explicit direction labels."""
    deltas: dict[str, float] = {}
    for metric in PRIMARY_METRICS:
        baseline_value = float(baseline_metrics[metric])
        cnv_value = float(cnv_metrics[metric])
        if not np.isfinite(baseline_value) or not np.isfinite(cnv_value):
            raise ValueError(f"paired metric {metric!r} must be finite")
        deltas[f"cnv_minus_baseline__{metric}"] = cnv_value - baseline_value
    deltas["improvement__mse"] = float(baseline_metrics["mse"]) - float(
        cnv_metrics["mse"]
    )
    deltas["improvement__mae"] = float(baseline_metrics["mae"]) - float(
        cnv_metrics["mae"]
    )
    deltas["improvement__residual_r2_vs_train_gene_mean"] = float(
        cnv_metrics["residual_r2_vs_train_gene_mean"]
    ) - float(baseline_metrics["residual_r2_vs_train_gene_mean"])
    return deltas


def summarize_paired_results(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize fold deltas and seed-level means without pseudo-replication tests."""
    if not records:
        raise ValueError("at least one paired result is required")
    expected_delta_keys = sorted(str(key) for key in records[0]["deltas"])
    by_seed: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        seed = int(record["seed"])
        deltas = record.get("deltas")
        if (
            not isinstance(deltas, Mapping)
            or sorted(str(key) for key in deltas) != expected_delta_keys
        ):
            raise ValueError("paired result delta schemas differ")
        by_seed.setdefault(seed, []).append(record)

    def describe(values: Sequence[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("summary inputs must be finite")
        return {
            "n": int(array.size),
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "median": float(np.median(array)),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    fold_level = {
        key: describe([float(record["deltas"][key]) for record in records])
        for key in expected_delta_keys
    }
    seed_means: dict[str, dict[str, float]] = {}
    for seed, seed_records in sorted(by_seed.items()):
        seed_means[str(seed)] = {
            key: float(
                np.mean([float(record["deltas"][key]) for record in seed_records])
            )
            for key in expected_delta_keys
        }
    seed_level = {
        key: describe([values[key] for values in seed_means.values()])
        for key in expected_delta_keys
    }
    return {
        "schema": "DepPLM-CNV-paired-summary-v1",
        "n_seed_fold_pairs": len(records),
        "n_seeds": len(by_seed),
        "fold_level_descriptive": fold_level,
        "seed_means": seed_means,
        "seed_level_descriptive": seed_level,
        "inferential_scope": (
            "Descriptive paired deltas only. Outer folds within a seed are not "
            "treated as independent biological replicates, and n=3 seeds is not "
            "used for a confirmatory significance claim."
        ),
    }


@dataclass
class _SensitivityArtifact:
    model: Any
    fold_data: Any
    config: dict[str, Any]
    gene_mean_reference: np.ndarray
    training: dict[str, Any]


def _relative_file_record(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"artifact path is outside the workspace: {resolved}")
    return {
        "path": resolved.relative_to(root_resolved).as_posix(),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _input_file_record(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Hash an input while retaining a portable path for workspace files."""
    resolved = path.resolve()
    root_resolved = root.resolve()
    display_path = (
        resolved.relative_to(root_resolved).as_posix()
        if resolved.is_relative_to(root_resolved)
        else str(resolved)
    )
    return {
        "path": display_path,
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _code_records() -> list[dict[str, Any]]:
    return [
        _relative_file_record(ROOT / relative)
        for relative in (
            "code/cnv_sensitivity.py",
            "code/corrected_protocol.py",
            "code/run_corrected_experiments.py",
            "code/models.py",
            "code/embedding_provenance.py",
        )
    ]


def _input_paths(cnv_path: Path) -> list[Path]:
    from embedding_provenance import (
        EMBEDDING_RELATIVE_PATH,
        MANIFEST_RELATIVE_PATH,
        NAMES_RELATIVE_PATH,
        NORMALIZED_RELATIVE_PATH,
    )

    return [
        cnv_path,
        PROCESSED / "gene_effect.npy",
        PROCESSED / "gene_seqs.fa",
        PROCESSED / "ppi_graph.npz",
        PROCESSED / "genes.txt",
        PROCESSED / "cell_lines.txt",
        PROCESSED / "lineages.txt",
        ROOT / EMBEDDING_RELATIVE_PATH,
        ROOT / NORMALIZED_RELATIVE_PATH,
        ROOT / NAMES_RELATIVE_PATH,
        ROOT / MANIFEST_RELATIVE_PATH,
        ROOT / "data" / "depmap" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        ROOT / "data" / "depmap" / "OmicsSomaticMutations.csv",
    ]


def _git_record() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _environment_record(device: str) -> dict[str, Any]:
    import scipy
    import sklearn
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "requested_device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "cudnn": int(torch.backends.cudnn.version())
        if torch.backends.cudnn.is_available()
        else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _fold_preprocessing_manifest(
    fold_data: Any,
    evaluation_indices: np.ndarray,
    *,
    cnv_fit: FittedCnvPCA | None,
    cnv_genes: Sequence[str],
    fit_indices: np.ndarray,
) -> dict[str, Any]:
    import run_corrected_experiments as runner

    record = runner.preprocessing_manifest(fold_data, evaluation_indices)
    expression_components = int(fold_data.expression_fit.n_output_components)
    record["expression"]["output_components"] = expression_components
    record["combined_cell_feature_components"] = int(fold_data.expression.shape[1])
    if cnv_fit is None:
        record["cnv"] = {
            "included": False,
            "components": 0,
            "fit_scope": None,
        }
    else:
        record["cnv"] = {
            "included": True,
            **cnv_fit_manifest(cnv_fit, cnv_genes, fit_indices),
        }
    return record


def _run_arm(
    *,
    arm: str,
    seed: int,
    fold_label: str,
    outer_train: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    outer_test: np.ndarray,
    base: Any,
    inner_data: Any,
    final_data: Any,
    inner_cnv_fit: FittedCnvPCA | None,
    final_cnv_fit: FittedCnvPCA | None,
    cnv_genes: Sequence[str],
    grid: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    device: str,
    run_dir: Path,
) -> dict[str, Any]:
    import torch
    import run_corrected_experiments as runner
    from corrected_protocol import fit_gene_mean_baseline, run_nested_selection

    configurations = [{"model": "depplm", **dict(candidate)} for candidate in grid]
    test_calls = 0
    prediction_record: dict[str, Any] | None = None

    def tune_fn(
        config: Mapping[str, Any],
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        run_seed: int,
    ):
        model = runner.build_model(config, base, inner_data)
        _, tuning = runner._train_epochs(
            model,
            config,
            base,
            inner_data,
            train_idx,
            epochs=args.inner_max_epochs,
            batch_size=args.batch_size,
            gene_subsample=args.gene_subsample,
            device=device,
            seed=run_seed,
            validation_idx=val_idx,
            patience=args.patience,
            scheduler_horizon=args.inner_max_epochs,
        )
        return tuning

    def refit_fn(
        config: Mapping[str, Any], train_idx: np.ndarray, epochs: int, run_seed: int
    ):
        if not np.array_equal(np.sort(train_idx), np.sort(outer_train)):
            raise RuntimeError(
                "refit indices differ from the planned outer training split"
            )
        gene_mean = fit_gene_mean_baseline(base.gene_effect, train_idx)
        model = runner.build_model(config, base, final_data)
        model, training = runner._train_epochs(
            model,
            config,
            base,
            final_data,
            train_idx,
            epochs=epochs,
            batch_size=args.batch_size,
            gene_subsample=args.gene_subsample,
            device=device,
            seed=run_seed,
            validation_idx=None,
            patience=args.patience,
            scheduler_horizon=args.inner_max_epochs,
        )
        return _SensitivityArtifact(
            model=model,
            fold_data=final_data,
            config=dict(config),
            gene_mean_reference=gene_mean,
            training=training,
        )

    def test_fn(artifact: _SensitivityArtifact, test_idx: np.ndarray):
        nonlocal prediction_record, test_calls
        test_calls += 1
        if test_calls != 1:
            raise RuntimeError("outer test was evaluated more than once for one arm")
        metrics, predictions = runner.evaluate_model(
            artifact.model,
            artifact.config,
            base,
            artifact.fold_data,
            test_idx,
            batch_size=args.batch_size,
            device=device,
            include_spearman=True,
            gene_mean_reference=artifact.gene_mean_reference,
            reference_scope="outer_train_gene_mean",
            return_predictions=True,
        )
        prediction_path = run_dir / f"{fold_label}-{arm}-predictions.npz"
        prediction_record = runner._atomic_write_outer_test_predictions(
            prediction_path,
            prediction=predictions,
            train_gene_mean=artifact.gene_mean_reference,
            outer_test_indices=test_idx,
            base=base,
        )
        return metrics

    started = time.time()
    selection = run_nested_selection(
        configs=configurations,
        outer_train_idx=outer_train,
        inner_train_idx=inner_train,
        inner_val_idx=inner_val,
        outer_test_idx=outer_test,
        tune_fn=tune_fn,
        refit_fn=refit_fn,
        test_fn=test_fn,
        seed=seed,
    )
    artifact = selection.final_model
    if not isinstance(artifact, _SensitivityArtifact) or prediction_record is None:
        raise RuntimeError("paired arm failed to produce its final model/predictions")
    record = {
        "schema": "DepPLM-CNV-arm-result-v1",
        "arm": arm,
        "uses_cnv": arm == "cnv",
        "best_config": dict(selection.best_config),
        "best_epoch": int(selection.best_epoch),
        "selection_score": float(selection.selection_score),
        "selection_metric": "residual_r2_vs_train_gene_mean",
        "tuning": selection.tuning_results,
        "optimization": {
            "scheduler": artifact.training["scheduler"],
            "scheduler_horizon_epochs": int(
                artifact.training["scheduler_horizon_epochs"]
            ),
            "refit_epochs": int(artifact.training["trained_epochs"]),
        },
        "test_metrics": dict(selection.test_metrics),
        "outer_test_predictions": prediction_record,
        "outer_test_evaluation_calls": test_calls,
        "preprocessing": {
            "inner_train_fit": _fold_preprocessing_manifest(
                inner_data,
                inner_val,
                cnv_fit=inner_cnv_fit,
                cnv_genes=cnv_genes,
                fit_indices=inner_train,
            ),
            "outer_train_refit": _fold_preprocessing_manifest(
                final_data,
                outer_test,
                cnv_fit=final_cnv_fit,
                cnv_genes=cnv_genes,
                fit_indices=outer_train,
            ),
        },
        "wall_seconds": float(time.time() - started),
    }
    del selection.final_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return record


def _canonical_reasons(args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if tuple(int(value) for value in args.seeds) != CANONICAL_SEEDS:
        reasons.append(f"seeds must be exactly {CANONICAL_SEEDS!r}")
    expected = {
        "folds": CANONICAL_FOLDS,
        "cnv_components": CANONICAL_CNV_COMPONENTS,
        "n_expr_pc": CANONICAL_EXPR_COMPONENTS,
        "inner_max_epochs": CANONICAL_INNER_EPOCHS,
        "patience": CANONICAL_PATIENCE,
        "batch_size": CANONICAL_BATCH_SIZE,
        "gene_subsample": CANONICAL_GENE_SUBSAMPLE,
        "top_mut": CANONICAL_TOP_MUT,
    }
    for field, value in expected.items():
        if int(getattr(args, field)) != value:
            reasons.append(f"{field} must be {value}")
    if int(args.fold_id) != -1:
        reasons.append("fold_id must be -1 so every outer fold is completed")
    normalized_grid = [
        {
            "hidden": int(value["hidden"]),
            "lr": float(value["lr"]),
            "wd": float(value["wd"]),
            "lamb_sel": float(value["lamb_sel"]),
        }
        for value in args.grid
    ]
    if normalized_grid != [dict(value) for value in CANONICAL_GRID]:
        reasons.append("the exact ordered four-point canonical grid is required")
    if not np.isclose(float(args.inner_val_fraction), 0.2, rtol=0.0, atol=1e-12):
        reasons.append("inner_val_fraction must be 0.2")
    return reasons


def run_paired_sensitivity(args: argparse.Namespace) -> Path:
    """Execute a hash-bound three-seed paired baseline/CNV experiment."""
    import torch
    import run_corrected_experiments as runner
    from embedding_provenance import validate_embedding_bundle
    from corrected_protocol import (
        atomic_write_json,
        build_split_manifest,
        derive_seed,
        make_inner_split,
        make_kfold_outer_splits,
        make_unique_run_dir,
    )

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(set(int(value) for value in args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")

    run_dir = make_unique_run_dir(
        args.output_root,
        protocol="paired-p1",
        condition="DepPLM-CNV",
        seed=int(args.seeds[0]),
    )
    launch_code = _code_records()
    started = time.time()
    status_path = run_dir / "run-manifest.json"
    try:
        embedding_provenance = validate_embedding_bundle(ROOT)
        print("[cnv] loading the full corrected DepPLM cohort", flush=True)
        source_base = runner.load_base_data(
            pilot=False,
            seed=int(args.seeds[0]),
            pilot_cells=0,
            pilot_genes=0,
        )
        print(
            "[cnv] aligning and loading the pinned 24Q4 absolute-CN matrix", flush=True
        )
        cnv = load_cnv_cohort(
            args.cnv_path,
            source_base.cell_ids,
            source_base.genes,
        )
        base = restrict_base_to_cnv_cohort(source_base, cnv)
        del source_base
        gc.collect()

        canonical_reasons = _canonical_reasons(args)
        requested_pairs = [
            {"seed": int(seed), "fold": int(fold)}
            for seed in args.seeds
            for fold in range(int(args.folds))
            if int(args.fold_id) in {-1, fold}
        ]
        manifest = {
            "schema": "DepPLM-CNV-sensitivity-run-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_status": "running",
            "scientific_result": False,
            "protocol": {
                "design": "paired_P1_outer_cross_validation",
                "model": "DepPLM",
                "arms": ["baseline", "baseline_plus_cnv"],
                "pairing": "identical_cells_outer_splits_and_inner_splits",
                "hyperparameter_selection": "independent_per_arm_inner_validation",
                "outer_test_access": "once_per_arm_after_fresh_refit",
                "cnv_preprocessing": (
                    "featurewise training-mean imputation, "
                    "training-only z scaling, training-only full-SVD PCA"
                ),
                "missing_source_rows": "excluded_before_both_arms_no_row_imputation",
                "seeds": [int(value) for value in args.seeds],
                "folds": int(args.folds),
                "cnv_components": int(args.cnv_components),
                "expression_components": int(args.n_expr_pc),
                "inner_validation_fraction": float(args.inner_val_fraction),
                "inner_max_epochs": int(args.inner_max_epochs),
                "patience": int(args.patience),
                "batch_size": int(args.batch_size),
                "gene_subsample": int(args.gene_subsample),
                "top_mutation_features": int(args.top_mut),
                "grid": [dict(value) for value in args.grid],
            },
            "canonical": not canonical_reasons,
            "canonical_exclusion_reasons": canonical_reasons,
            "requested_seed_fold_pairs": requested_pairs,
            "cohort": cnv.manifest(),
            "paired_cell_ids_sha256": _hash_strings(base.cell_ids),
            "paired_gene_ids_sha256": _hash_strings(base.genes),
            "paired_source_cell_indices_sha256": _hash_array(
                base.source_cell_indices.astype("<i8", copy=False)
            ),
            "paired_source_gene_indices_sha256": _hash_array(
                base.source_gene_indices.astype("<i8", copy=False)
            ),
            "code": launch_code,
            "inputs": [
                _input_file_record(path) for path in _input_paths(Path(args.cnv_path))
            ],
            "embedding_provenance": embedding_provenance,
            "environment": _environment_record(device),
            "git": _git_record(),
        }
        atomic_write_json(status_path, manifest)

        paired_results: list[dict[str, Any]] = []
        for seed_value in args.seeds:
            seed = int(seed_value)
            splits = make_kfold_outer_splits(len(base.cell_ids), int(args.folds), seed)
            for fold_number, (outer_train, outer_test) in enumerate(splits):
                if int(args.fold_id) not in {-1, fold_number}:
                    continue
                fold_label = f"s{seed}-fold-{fold_number:02d}"
                fold_seed = derive_seed(seed, fold_label)
                inner_train, inner_val = make_inner_split(
                    outer_train,
                    base.lineages,
                    val_fraction=float(args.inner_val_fraction),
                    seed=derive_seed(fold_seed, "inner-split"),
                )
                split = build_split_manifest(
                    outer_train_idx=outer_train,
                    inner_train_idx=inner_train,
                    inner_val_idx=inner_val,
                    outer_test_idx=outer_test,
                    cell_ids=base.cell_ids,
                    seed=fold_seed,
                )
                split.update(
                    {
                        "schema": "DepPLM-CNV-paired-split-v1",
                        "seed": seed,
                        "fold": fold_number,
                        "pairing": paired_split_manifest(
                            base_to_paired_view(base, cnv), outer_train, outer_test
                        ),
                        "source_cell_indices": {
                            "outer_train": [
                                int(base.source_cell_indices[index])
                                for index in outer_train
                            ],
                            "outer_test": [
                                int(base.source_cell_indices[index])
                                for index in outer_test
                            ],
                        },
                    }
                )
                split_path = run_dir / f"{fold_label}-split.json"
                atomic_write_json(split_path, split)
                split_record = _relative_file_record(split_path, root=run_dir)

                print(
                    f"[cnv] {fold_label}: fitting shared baseline preprocessing",
                    flush=True,
                )
                inner_baseline = runner.fit_fold_data(
                    base,
                    inner_train,
                    seed=derive_seed(fold_seed, "inner-preprocess"),
                    n_expr_pc=int(args.n_expr_pc),
                    top_mut=int(args.top_mut),
                )
                final_baseline = runner.fit_fold_data(
                    base,
                    outer_train,
                    seed=derive_seed(fold_seed, "outer-preprocess"),
                    n_expr_pc=int(args.n_expr_pc),
                    top_mut=int(args.top_mut),
                )
                inner_cnv, inner_cnv_fit = augment_fold_data_with_cnv(
                    inner_baseline,
                    cnv.values,
                    inner_train,
                    n_components=int(args.cnv_components),
                    seed=derive_seed(fold_seed, "inner-cnv-pca"),
                )
                final_cnv, final_cnv_fit = augment_fold_data_with_cnv(
                    final_baseline,
                    cnv.values,
                    outer_train,
                    n_components=int(args.cnv_components),
                    seed=derive_seed(fold_seed, "outer-cnv-pca"),
                )

                baseline_record = _run_arm(
                    arm="baseline",
                    seed=fold_seed,
                    fold_label=fold_label,
                    outer_train=outer_train,
                    inner_train=inner_train,
                    inner_val=inner_val,
                    outer_test=outer_test,
                    base=base,
                    inner_data=inner_baseline,
                    final_data=final_baseline,
                    inner_cnv_fit=None,
                    final_cnv_fit=None,
                    cnv_genes=cnv.gene_names,
                    grid=args.grid,
                    args=args,
                    device=device,
                    run_dir=run_dir,
                )
                cnv_record = _run_arm(
                    arm="cnv",
                    seed=fold_seed,
                    fold_label=fold_label,
                    outer_train=outer_train,
                    inner_train=inner_train,
                    inner_val=inner_val,
                    outer_test=outer_test,
                    base=base,
                    inner_data=inner_cnv,
                    final_data=final_cnv,
                    inner_cnv_fit=inner_cnv_fit,
                    final_cnv_fit=final_cnv_fit,
                    cnv_genes=cnv.gene_names,
                    grid=args.grid,
                    args=args,
                    device=device,
                    run_dir=run_dir,
                )
                pair_record = {
                    "schema": "DepPLM-CNV-paired-fold-result-v1",
                    "seed": seed,
                    "fold": fold_number,
                    "fold_seed": fold_seed,
                    "n_train": int(outer_train.size),
                    "n_test": int(outer_test.size),
                    "split_manifest": split_record,
                    "baseline": baseline_record,
                    "cnv": cnv_record,
                    "deltas": paired_metric_deltas(
                        baseline_record["test_metrics"], cnv_record["test_metrics"]
                    ),
                }
                atomic_write_json(
                    run_dir / f"{fold_label}-paired-result.json", pair_record
                )
                paired_results.append(pair_record)

        expected_count = len(requested_pairs)
        code_at_completion = _code_records()
        result = {
            "schema": "DepPLM-CNV-sensitivity-result-v1",
            "paired_results": paired_results,
            "summary": summarize_paired_results(paired_results),
            "n_completed_seed_fold_pairs": len(paired_results),
            "n_expected_seed_fold_pairs": expected_count,
            "wall_seconds": float(time.time() - started),
        }
        atomic_write_json(run_dir / "results.json", result)
        execution_reasons: list[str] = []
        if len(paired_results) != expected_count:
            execution_reasons.append(
                f"completed {len(paired_results)} of {expected_count} seed-fold pairs"
            )
        if launch_code != code_at_completion:
            execution_reasons.append("code hashes changed during execution")
        execution_complete = not execution_reasons
        protocol_complete = execution_complete and not canonical_reasons
        completion_reasons = [*execution_reasons, *canonical_reasons]
        completion = {
            "schema": "DepPLM-CNV-sensitivity-completion-v1",
            "run_status": "complete" if execution_complete else "partial",
            "execution_complete": execution_complete,
            "protocol_complete": protocol_complete,
            "eligible_for_independent_verification": execution_complete,
            "scientific_result": False,
            "scientific_result_reason": (
                "Predictions exist and the canonical paired protocol completed, but "
                "an independent verify command must recompute metrics from artifacts."
                if protocol_complete
                else (
                    "The requested non-canonical run completed and may be verified, "
                    "but it is not eligible as the canonical CNV sensitivity result."
                    if execution_complete
                    else "; ".join(completion_reasons)
                )
            ),
            "completion_reasons": completion_reasons,
            "code_at_launch": launch_code,
            "code_at_completion": code_at_completion,
            "results": _relative_file_record(run_dir / "results.json"),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(run_dir / "completion.json", completion)
        return run_dir
    except Exception as exc:
        failure = {
            "schema": "DepPLM-CNV-sensitivity-failure-v1",
            "run_status": "failed",
            "scientific_result": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            from corrected_protocol import atomic_write_json

            atomic_write_json(run_dir / "failure.json", failure)
        except Exception:
            pass
        raise


def base_to_paired_view(base: Any, cnv: CnvCohort) -> PairedCnvCohort:
    """Build the identity-only paired view used by split validation."""
    if tuple(base.cell_ids) != cnv.cell_ids:
        raise ValueError("restricted baseline and CNV cohorts are not identical")
    return PairedCnvCohort(
        cell_ids=tuple(base.cell_ids),
        original_cell_indices=cnv.coverage.request_cell_indices.copy(),
        baseline_values=np.empty((len(base.cell_ids), 0), dtype=np.float32),
        cnv_values=cnv.values,
        cnv_gene_names=cnv.gene_names,
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_prediction_artifact(
    *,
    run_dir: Path,
    record: Mapping[str, Any],
    split: Mapping[str, Any],
    all_cells: Sequence[str],
    all_genes: Sequence[str],
    gene_effect: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from corrected_protocol import dependency_metrics, fit_gene_mean_baseline
    from scipy.stats import rankdata

    required_arrays = {
        "prediction",
        "train_gene_mean",
        "outer_test_indices",
        "source_cell_indices",
        "source_gene_indices",
        "cell_ids",
        "gene_ids",
    }
    relative = Path(str(record["path"]))
    if relative.is_absolute():
        raise ValueError("prediction record path must be relative to its run directory")
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError("prediction record escapes its run directory")
    current = _relative_file_record(path)
    if (
        int(record["bytes"]) != current["bytes"]
        or str(record["sha256"]) != current["sha256"]
    ):
        raise ValueError(f"prediction file hash/size mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required_arrays:
            raise ValueError(
                f"unexpected prediction arrays in {path}: {archive.files!r}"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    stored_arrays = record.get("arrays")
    actual_arrays = {
        name: {"dtype": str(array.dtype), "shape": list(array.shape)}
        for name, array in arrays.items()
    }
    if stored_arrays != actual_arrays:
        raise ValueError(f"prediction array metadata mismatch: {path}")

    outer_test = np.asarray(split["outer_test"]["indices"], dtype=np.int64)
    if not np.array_equal(arrays["outer_test_indices"], outer_test):
        raise ValueError("prediction outer-test indices differ from split manifest")
    source_cells = arrays["source_cell_indices"].astype(np.int64, copy=False)
    source_genes = arrays["source_gene_indices"].astype(np.int64, copy=False)
    if source_cells.ndim != 1 or source_genes.ndim != 1:
        raise ValueError("source index arrays must be one-dimensional")
    expected_source_cells = np.asarray(
        split["source_cell_indices"]["outer_test"], dtype=np.int64
    )
    if not np.array_equal(source_cells, expected_source_cells):
        raise ValueError("prediction source cells differ from the split manifest")
    if source_cells.size and (
        int(source_cells.min()) < 0 or int(source_cells.max()) >= len(all_cells)
    ):
        raise IndexError("prediction source cell index is outside cell_lines.txt")
    if source_genes.size and (
        int(source_genes.min()) < 0 or int(source_genes.max()) >= len(all_genes)
    ):
        raise IndexError("prediction source gene index is outside genes.txt")
    expected_cells = np.asarray(
        [all_cells[int(index)] for index in source_cells], dtype=str
    )
    expected_genes = np.asarray(
        [all_genes[int(index)] for index in source_genes], dtype=str
    )
    if not np.array_equal(arrays["cell_ids"].astype(str), expected_cells):
        raise ValueError("prediction cell IDs do not match source_cell_indices")
    if not np.array_equal(arrays["gene_ids"].astype(str), expected_genes):
        raise ValueError("prediction gene IDs do not match source_gene_indices")

    train_source = np.asarray(
        split["source_cell_indices"]["outer_train"], dtype=np.int64
    )
    expected_mean = fit_gene_mean_baseline(
        np.asarray(gene_effect[np.ix_(source_genes, train_source)], dtype=np.float32),
        np.arange(train_source.size, dtype=np.int64),
    )
    if not np.array_equal(arrays["train_gene_mean"], expected_mean, equal_nan=True):
        raise ValueError(
            "stored training gene mean does not match the split-bound target"
        )
    target = np.asarray(
        gene_effect[np.ix_(source_genes, source_cells)], dtype=np.float32
    ).T
    prediction = arrays["prediction"].astype(np.float32, copy=False)
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("prediction shape/finiteness is invalid")
    metrics = dependency_metrics(prediction, target, arrays["train_gene_mean"])
    spearman: list[float] = []
    for row in range(prediction.shape[0]):
        finite = np.isfinite(prediction[row]) & np.isfinite(target[row])
        if int(finite.sum()) < 10:
            continue
        ranked_prediction = rankdata(prediction[row, finite])
        ranked_target = rankdata(target[row, finite])
        if np.std(ranked_prediction) <= 0 or np.std(ranked_target) <= 0:
            continue
        spearman.append(float(np.corrcoef(ranked_prediction, ranked_target)[0, 1]))
    if not spearman:
        raise ValueError("prediction artifact has no finite cellwise Spearman values")
    values = np.asarray(spearman, dtype=np.float64)
    metrics.update(
        {
            "cellwise_spearman_mean": float(values.mean()),
            "cellwise_spearman_sd": float(values.std()),
            "n_finite_cellwise_spearman": int(values.size),
        }
    )
    arrays["target"] = target
    return metrics, arrays


def _assert_metrics_match(
    stored: Mapping[str, Any], recomputed: Mapping[str, Any], *, label: str
) -> None:
    for metric, expected in recomputed.items():
        if metric not in stored:
            raise ValueError(f"{label} is missing recomputable metric {metric!r}")
        observed = stored[metric]
        if isinstance(expected, int):
            if int(observed) != expected:
                raise ValueError(f"{label} metric {metric!r} does not match")
        elif not np.isclose(float(observed), float(expected), rtol=1e-10, atol=1e-12):
            raise ValueError(f"{label} metric {metric!r} does not match")


def _assert_summary_match(
    stored: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> None:
    """Compare summary structure exactly and floats within the audit tolerance."""
    for field in ("schema", "n_seed_fold_pairs", "n_seeds", "inferential_scope"):
        if stored.get(field) != recomputed.get(field):
            raise ValueError(f"paired summary field {field!r} differs")
    for section in ("fold_level_descriptive", "seed_level_descriptive"):
        stored_section = stored.get(section)
        recomputed_section = recomputed.get(section)
        if not isinstance(stored_section, Mapping) or not isinstance(
            recomputed_section, Mapping
        ):
            raise ValueError(f"paired summary section {section!r} is invalid")
        if set(stored_section) != set(recomputed_section):
            raise ValueError(f"paired summary section {section!r} keys differ")
        for metric in stored_section:
            _assert_metrics_match(
                stored_section[metric],
                recomputed_section[metric],
                label=f"summary/{section}/{metric}",
            )
    stored_seed_means = stored.get("seed_means")
    recomputed_seed_means = recomputed.get("seed_means")
    if not isinstance(stored_seed_means, Mapping) or not isinstance(
        recomputed_seed_means, Mapping
    ):
        raise ValueError("paired summary seed means are invalid")
    if set(stored_seed_means) != set(recomputed_seed_means):
        raise ValueError("paired summary seed keys differ")
    for seed in stored_seed_means:
        _assert_metrics_match(
            stored_seed_means[seed],
            recomputed_seed_means[seed],
            label=f"summary/seed_means/{seed}",
        )


def _verify_file_record(
    record: Mapping[str, Any], *, base: Path, require_within_base: bool
) -> Path:
    relative = Path(str(record["path"]))
    path = relative.resolve() if relative.is_absolute() else (base / relative).resolve()
    if require_within_base and not path.is_relative_to(base.resolve()):
        raise ValueError(f"artifact path escapes its run directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if int(record["bytes"]) != int(path.stat().st_size):
        raise ValueError(f"file size mismatch: {path}")
    if str(record["sha256"]) != _sha256_file(path):
        raise ValueError(f"file hash mismatch: {path}")
    return path


def verify_cnv_run(run_dir: str | Path) -> dict[str, Any]:
    """Re-hash artifacts and independently recompute every outer-test metric."""
    from corrected_protocol import (
        derive_seed,
        make_inner_split,
        make_kfold_outer_splits,
    )
    from embedding_provenance import validate_embedding_bundle

    run_path = Path(run_dir).resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(run_path)
    manifest = _load_json_object(run_path / "run-manifest.json")
    completion = _load_json_object(run_path / "completion.json")
    results_path = run_path / "results.json"
    results = _load_json_object(results_path)
    if not bool(completion.get("eligible_for_independent_verification")):
        raise ValueError("CNV run did not complete its requested experiment")
    current_result_record = _relative_file_record(results_path)
    stored_result_record = completion.get("results", {})
    if current_result_record["bytes"] != stored_result_record.get(
        "bytes"
    ) or current_result_record["sha256"] != stored_result_record.get("sha256"):
        raise ValueError("results.json hash/size differs from completion.json")
    if manifest.get("code") != completion.get("code_at_launch"):
        raise ValueError("launch code records differ between manifests")
    if completion.get("code_at_launch") != completion.get("code_at_completion"):
        raise ValueError("code changed while the CNV run was executing")
    if _code_records() != completion.get("code_at_completion"):
        raise ValueError("current code hashes differ from the executed CNV code")

    verified_inputs = [
        _verify_file_record(record, base=ROOT, require_within_base=False)
        for record in manifest.get("inputs", [])
    ]
    if len(verified_inputs) != len(_input_paths(DEFAULT_CNV_PATH)):
        raise ValueError("run manifest has an incomplete input inventory")
    cnv_paths = [path for path in verified_inputs if path.name == CNV_FILENAME]
    if len(cnv_paths) != 1:
        raise ValueError("run manifest must bind exactly one official CNV source")
    expected_inputs = {path.resolve() for path in _input_paths(cnv_paths[0])}
    if set(verified_inputs) != expected_inputs:
        raise ValueError("run manifest input paths differ from the complete inventory")
    embedding_provenance = validate_embedding_bundle(ROOT)
    if embedding_provenance != manifest.get("embedding_provenance"):
        raise ValueError("current ESM provenance differs from the executed run")

    all_cells = (PROCESSED / "cell_lines.txt").read_text(encoding="utf-8").splitlines()
    all_genes = (PROCESSED / "genes.txt").read_text(encoding="utf-8").splitlines()
    all_lineages = np.asarray(
        (PROCESSED / "lineages.txt").read_text(encoding="utf-8").splitlines(),
        dtype=str,
    )
    gene_effect = np.load(PROCESSED / "gene_effect.npy", mmap_mode="r")
    cnv = load_cnv_cohort(cnv_paths[0], all_cells, all_genes)
    if json.dumps(cnv.manifest(), sort_keys=True) != json.dumps(
        manifest.get("cohort"), sort_keys=True
    ):
        raise ValueError("current CNV cohort differs from the executed run")
    paired_cell_ids = cnv.cell_ids
    paired_lineages = all_lineages[cnv.coverage.request_cell_indices]
    if _hash_strings(paired_cell_ids) != manifest.get("paired_cell_ids_sha256"):
        raise ValueError("paired cell identity hash differs from the run manifest")
    if _hash_strings(all_genes) != manifest.get("paired_gene_ids_sha256"):
        raise ValueError("paired output-gene hash differs from the run manifest")
    if _hash_array(
        cnv.coverage.request_cell_indices.astype("<i8", copy=False)
    ) != manifest.get("paired_source_cell_indices_sha256"):
        raise ValueError("paired source-cell index hash differs from the run manifest")
    expected_gene_indices = np.arange(len(all_genes), dtype=np.int64)
    if _hash_array(expected_gene_indices.astype("<i8", copy=False)) != manifest.get(
        "paired_source_gene_indices_sha256"
    ):
        raise ValueError("paired source-gene index hash differs from the run manifest")

    protocol = manifest.get("protocol", {})
    expected_splits: dict[tuple[int, int], tuple[np.ndarray, ...]] = {}
    for seed_value in protocol.get("seeds", []):
        seed = int(seed_value)
        outer_splits = make_kfold_outer_splits(
            len(paired_cell_ids), int(protocol["folds"]), seed
        )
        for fold, (outer_train, outer_test) in enumerate(outer_splits):
            fold_seed = derive_seed(seed, f"s{seed}-fold-{fold:02d}")
            inner_train, inner_val = make_inner_split(
                outer_train,
                paired_lineages,
                val_fraction=float(protocol["inner_validation_fraction"]),
                seed=derive_seed(fold_seed, "inner-split"),
            )
            expected_splits[(seed, fold)] = (
                outer_train,
                inner_train,
                inner_val,
                outer_test,
            )

    verified_records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for pair in results.get("paired_results", []):
        seed = int(pair["seed"])
        fold = int(pair["fold"])
        key = (seed, fold)
        if key in seen:
            raise ValueError(f"duplicate CNV seed/fold pair {key!r}")
        seen.add(key)
        split_path = _verify_file_record(
            pair["split_manifest"], base=run_path, require_within_base=True
        )
        split = _load_json_object(split_path)
        if key not in expected_splits:
            raise ValueError(f"unexpected seed/fold pair {key!r}")
        (
            expected_outer_train,
            expected_inner_train,
            expected_inner_val,
            expected_outer_test,
        ) = expected_splits[key]
        for field, expected_indices in (
            ("outer_train", expected_outer_train),
            ("inner_train", expected_inner_train),
            ("inner_validation", expected_inner_val),
            ("outer_test", expected_outer_test),
        ):
            observed_indices = np.asarray(split[field]["indices"], dtype=np.int64)
            if not np.array_equal(observed_indices, expected_indices):
                raise ValueError(
                    f"split {field} differs from deterministic protocol for {key!r}"
                )
        expected_source_train = cnv.coverage.request_cell_indices[expected_outer_train]
        expected_source_test = cnv.coverage.request_cell_indices[expected_outer_test]
        if not np.array_equal(
            np.asarray(split["source_cell_indices"]["outer_train"], dtype=np.int64),
            expected_source_train,
        ) or not np.array_equal(
            np.asarray(split["source_cell_indices"]["outer_test"], dtype=np.int64),
            expected_source_test,
        ):
            raise ValueError(f"split source-cell mapping differs for {key!r}")
        expected_pairing = paired_split_manifest(
            PairedCnvCohort(
                cell_ids=paired_cell_ids,
                original_cell_indices=cnv.coverage.request_cell_indices,
                baseline_values=np.empty((len(paired_cell_ids), 0), dtype=np.float32),
                cnv_values=cnv.values,
                cnv_gene_names=cnv.gene_names,
            ),
            expected_outer_train,
            expected_outer_test,
        )
        if split.get("pairing") != expected_pairing:
            raise ValueError(f"paired split binding differs for {key!r}")
        if not bool(split.get("pairing", {}).get("baseline_cnv_cell_ids_identical")):
            raise ValueError(f"paired identity gate failed for {key!r}")
        recomputed: dict[str, dict[str, Any]] = {}
        arrays_by_arm: dict[str, dict[str, np.ndarray]] = {}
        for arm in ("baseline", "cnv"):
            arm_record = pair[arm]
            metrics, arrays = _validate_prediction_artifact(
                run_dir=run_path,
                record=arm_record["outer_test_predictions"],
                split=split,
                all_cells=all_cells,
                all_genes=all_genes,
                gene_effect=gene_effect,
            )
            _assert_metrics_match(
                arm_record["test_metrics"], metrics, label=f"{key!r}/{arm}"
            )
            recomputed[arm] = metrics
            arrays_by_arm[arm] = arrays
        for array_name in (
            "outer_test_indices",
            "source_cell_indices",
            "source_gene_indices",
            "cell_ids",
            "gene_ids",
            "target",
        ):
            left = arrays_by_arm["baseline"][array_name]
            right = arrays_by_arm["cnv"][array_name]
            equal = (
                np.array_equal(left, right, equal_nan=True)
                if np.issubdtype(left.dtype, np.floating)
                else np.array_equal(left, right)
            )
            if not equal:
                raise ValueError(f"paired arms differ in {array_name} for {key!r}")
        deltas = paired_metric_deltas(recomputed["baseline"], recomputed["cnv"])
        _assert_metrics_match(pair["deltas"], deltas, label=f"{key!r}/deltas")
        verified_records.append({"seed": seed, "fold": fold, "deltas": deltas})

    requested = {
        (int(value["seed"]), int(value["fold"]))
        for value in manifest.get("requested_seed_fold_pairs", [])
    }
    if seen != requested:
        raise ValueError("verified seed/fold pairs do not match the run manifest")
    recomputed_summary = summarize_paired_results(verified_records)
    stored_summary = results.get("summary")
    if not isinstance(stored_summary, Mapping):
        raise ValueError("stored paired summary is invalid")
    _assert_summary_match(stored_summary, recomputed_summary)

    canonical_pairs = {
        (seed, fold) for seed in CANONICAL_SEEDS for fold in range(CANONICAL_FOLDS)
    }
    scientific_result = (
        bool(manifest.get("canonical"))
        and seen == requested
        and seen == canonical_pairs
        and len(verified_records) == len(canonical_pairs)
    )
    report = {
        "schema": "DepPLM-CNV-sensitivity-verification-v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": run_path.name,
        "execution_complete": bool(completion.get("execution_complete")),
        "protocol_complete": bool(completion.get("protocol_complete")),
        "artifacts_hash_verified": True,
        "outer_test_metrics_recomputed": True,
        "paired_targets_and_identities_identical": True,
        "n_verified_seed_fold_pairs": len(verified_records),
        "input_files_hash_verified": True,
        "embedding_provenance_verified": True,
        "cnv_cohort_reloaded_and_verified": True,
        "deterministic_splits_recomputed": True,
        "scientific_result": scientific_result,
        "scientific_result_reason": (
            "Canonical paired CNV protocol and all hash/metric/identity gates passed."
            if scientific_result
            else "Run is internally valid but does not use the canonical protocol."
        ),
        "result_sha256": current_result_record["sha256"],
        "summary": recomputed_summary,
    }
    return report


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Write one immutable compressed NumPy artifact and return its file record."""
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    temporary = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(
                handle,
                **{
                    str(name): np.ascontiguousarray(value)
                    for name, value in arrays.items()
                },
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing output: {target}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _relative_file_record(target, root=target.parent)


def write_coverage_manifest(
    *, cnv_path: Path, output_root: Path = DEFAULT_OUTPUT_ROOT
) -> Path:
    """Reload the official source and persist an auditable cohort-alignment record."""
    from corrected_protocol import atomic_write_json, make_unique_run_dir

    requested_cells = (
        (PROCESSED / "cell_lines.txt").read_text(encoding="utf-8").splitlines()
    )
    requested_genes = (PROCESSED / "genes.txt").read_text(encoding="utf-8").splitlines()
    cohort = load_cnv_cohort(cnv_path, requested_cells, requested_genes)
    run_dir = make_unique_run_dir(
        output_root, protocol="coverage", condition="CNV-24Q4", seed=0
    )
    alignment = _atomic_write_npz(
        run_dir / "alignment.npz",
        {
            "request_cell_indices": cohort.coverage.request_cell_indices.astype(
                np.int64, copy=False
            ),
            "request_gene_indices": cohort.coverage.request_gene_indices.astype(
                np.int64, copy=False
            ),
            "source_row_indices": cohort.coverage.source_row_indices.astype(
                np.int64, copy=False
            ),
            "source_column_indices": cohort.coverage.source_column_indices.astype(
                np.int64, copy=False
            ),
            "matched_cell_ids": np.asarray(cohort.cell_ids, dtype=str),
            "matched_gene_ids": np.asarray(cohort.gene_names, dtype=str),
            "missing_cell_ids": np.asarray(cohort.coverage.missing_cell_ids, dtype=str),
            "missing_gene_ids": np.asarray(cohort.coverage.missing_genes, dtype=str),
        },
    )
    manifest = {
        "schema": "DepPLM-CNV-coverage-artifact-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_result": False,
        "scientific_result_reason": (
            "This artifact validates source identity and cohort coverage only; it "
            "does not contain model comparisons."
        ),
        "source_release": {
            "filename": CNV_FILENAME,
            "doi": CNV_RELEASE_DOI,
            "download_url": CNV_SOURCE_URL,
        },
        "cohort": cohort.manifest(),
        "alignment": alignment,
        "missing_cell_ids": list(cohort.coverage.missing_cell_ids),
        "missing_gene_ids": list(cohort.coverage.missing_genes),
        "inputs": [
            _input_file_record(cnv_path),
            _input_file_record(PROCESSED / "cell_lines.txt"),
            _input_file_record(PROCESSED / "genes.txt"),
        ],
        "code": [_relative_file_record(Path(__file__).resolve())],
    }
    atomic_write_json(run_dir / "coverage-manifest.json", manifest)
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-resistant paired DepPLM CNV sensitivity analysis"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    coverage = subparsers.add_parser(
        "coverage", help="validate the pinned source and persist cohort alignment"
    )
    coverage.add_argument("--cnv-path", type=Path, default=DEFAULT_CNV_PATH)
    coverage.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    run = subparsers.add_parser(
        "run", help="run paired baseline versus baseline+CNV nested evaluation"
    )
    run.add_argument("--cnv-path", type=Path, default=DEFAULT_CNV_PATH)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--seeds", type=int, nargs="+", default=list(CANONICAL_SEEDS))
    run.add_argument("--folds", type=int, default=CANONICAL_FOLDS)
    run.add_argument(
        "--fold-id",
        type=int,
        default=-1,
        help="-1 runs all folds; any single-fold run is non-canonical",
    )
    run.add_argument("--cnv-components", type=int, default=CANONICAL_CNV_COMPONENTS)
    run.add_argument("--n-expr-pc", type=int, default=CANONICAL_EXPR_COMPONENTS)
    run.add_argument("--inner-val-fraction", type=float, default=0.2)
    run.add_argument("--inner-max-epochs", type=int, default=CANONICAL_INNER_EPOCHS)
    run.add_argument("--patience", type=int, default=CANONICAL_PATIENCE)
    run.add_argument("--batch-size", type=int, default=CANONICAL_BATCH_SIZE)
    run.add_argument("--gene-subsample", type=int, default=CANONICAL_GENE_SUBSAMPLE)
    run.add_argument("--top-mut", type=int, default=CANONICAL_TOP_MUT)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument(
        "--grid-profile",
        choices=("canonical", "single"),
        default="canonical",
        help="single is a non-canonical smoke-test profile",
    )

    verify = subparsers.add_parser(
        "verify", help="re-hash artifacts and recompute metrics from predictions"
    )
    verify.add_argument("run_dir", type=Path)

    args = parser.parse_args(argv)
    if args.command == "run":
        args.grid = (
            [dict(value) for value in CANONICAL_GRID]
            if args.grid_profile == "canonical"
            else [dict(CANONICAL_GRID[0])]
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "coverage":
        run_dir = write_coverage_manifest(
            cnv_path=args.cnv_path, output_root=args.output_root
        )
        print(run_dir)
        return 0
    if args.command == "run":
        run_dir = run_paired_sensitivity(args)
        print(run_dir)
        return 0
    if args.command == "verify":
        from corrected_protocol import atomic_write_json

        report = verify_cnv_run(args.run_dir)
        output = args.run_dir.resolve() / "verification.json"
        atomic_write_json(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
