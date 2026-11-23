#!/usr/bin/env python3
"""Run the registered DepPLM protein-family holdout experiment.

This protocol holds out complete sequence-connected components over genes.  It
does not hold out cell lines.  All cell covariates and all frozen gene features
are available, while every gene-effect label belonging to the outer test
families is inaccessible until the single final evaluation call.

The main split is the registered MMseqs2 30% identity / 80% bidirectional
coverage split.  For outer fold ``i``, fold ``(i + 1) mod 5`` is the inner
validation set and the other three folds are inner training genes.  After
hyperparameter and epoch selection, a fresh model is fit on all four outer
training folds and evaluated once on the held-out family fold.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import rankdata

from corrected_protocol import (
    K_TOKENS,
    AtomicWriteError,
    derive_seed,
    fit_comutation_edge_weights,
    fit_expression_pca,
    fit_lineage_one_hot,
    fit_mutation_features,
    set_global_seed,
)
from embedding_provenance import (
    EMBEDDING_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    NAMES_RELATIVE_PATH,
    NORMALIZED_RELATIVE_PATH,
    validate_embedding_bundle,
)
from models import normalized_adjacency
from family_holdout import assign_components_to_folds
from run_corrected_experiments import (
    BaseData,
    FoldData,
    _forward,
    build_model,
    load_base_data,
)


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DEPMAP = ROOT / "data" / "depmap"
FAMILY_ROOT = ROOT / "results" / "family_holdout_v1"
SPLIT_DIR = FAMILY_ROOT / "identity_30"
OUTPUT_ROOT = FAMILY_ROOT / "model_evaluation_v1"

SCHEMA = "DepPLM-family-holdout-model-evaluation-v2"
PREDICTION_SCHEMA = "DepPLM-family-holdout-outer-predictions-v2"
MAIN_IDENTITY_PERCENT = 30.0
MIN_BIDIRECTIONAL_COVERAGE = 0.80
N_FOLDS = 5
REGISTERED_SPLIT_SEEDS = (42, 43, 44)
REGISTERED_TRAINING_SEEDS = (42, 43, 44)
NEURAL_CONDITIONS: dict[str, dict[str, Any]] = {
    "DepPLM": {"model": "depplm"},
    "C-PLM-MLP": {"model": "plmmlp"},
}
BASELINE_CONDITION = "TRAIN-GENE-CELL-MEAN"
ALL_CONDITIONS = (*NEURAL_CONDITIONS, BASELINE_CONDITION)
FULL_GRID = [
    {"hidden": hidden, "lr": lr, "wd": 1e-5, "lamb_sel": 0.1}
    for hidden in (128, 256)
    for lr in (1e-3, 3e-4)
]
CANONICAL_ARGUMENTS: dict[str, int] = {
    "inner_max_epochs": 20,
    "patience": 5,
    "batch_size": 64,
    "gene_subsample": 6_000,
    "top_mut": 5_000,
    "n_expr_pc": 128,
}


class FamilyExperimentError(RuntimeError):
    """Raised when a family-holdout run cannot be trusted."""


@dataclass(frozen=True)
class FamilyAssignment:
    gene: str
    component_id: str
    fold: int
    stratum: str


@dataclass(frozen=True)
class FamilySplit:
    outer_fold: int
    inner_validation_fold: int
    outer_train: np.ndarray
    outer_test: np.ndarray
    inner_train: np.ndarray
    inner_validation: np.ndarray


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
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


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: Path, *, relative_to: Path = ROOT) -> dict[str, Any]:
    path = path.resolve(strict=True)
    try:
        display = path.relative_to(relative_to.resolve(strict=True)).as_posix()
    except ValueError:
        display = str(path)
    stat = path.stat()
    return {
        "path": display,
        "bytes": int(stat.st_size),
        "sha256": _sha256_file(path),
    }


def _json_bytes(value: Any) -> bytes:
    value = _json_safe(value)
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _json_safe(value: Any) -> Any:
    """Represent undefined numerical summaries explicitly as JSON null."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
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


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    normalized = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    for name, value in normalized.items():
        if value.dtype == object:
            raise ValueError(f"object arrays are forbidden in prediction artifact: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    record = _file_record(path, relative_to=path.parent)
    return {
        "schema": PREDICTION_SCHEMA,
        **record,
        "arrays": {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _hash_array(value),
            }
            for name, value in sorted(normalized.items())
        },
    }


def validate_family_release(root: Path = FAMILY_ROOT) -> dict[str, Any]:
    """Fail closed unless the complete registered sequence-family release matches."""
    manifest_path = root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "depplm-family-holdout-v1":
        raise FamilyExperimentError("unexpected family release schema")
    if manifest.get("status") != "complete":
        raise FamilyExperimentError("family release is not complete")
    if not math.isclose(
        float(manifest.get("main_identity_percentage", -1.0)),
        MAIN_IDENTITY_PERCENT,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise FamilyExperimentError("family release main identity threshold changed")
    records: dict[str, dict[str, Any]] = {}
    for raw in manifest.get("files", []):
        relative = str(raw.get("path", "")).replace("\\", "/")
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise FamilyExperimentError(f"unsafe family release path: {relative!r}")
        if relative in records:
            raise FamilyExperimentError(f"duplicate family release path: {relative}")
        target = (root / Path(relative)).resolve(strict=True)
        try:
            target.relative_to(root.resolve(strict=True))
        except ValueError as exc:
            raise FamilyExperimentError(f"family release path escapes root: {relative}") from exc
        if target.stat().st_size != int(raw.get("bytes", -1)):
            raise FamilyExperimentError(f"family release byte count mismatch: {relative}")
        if _sha256_file(target) != raw.get("sha256"):
            raise FamilyExperimentError(f"family release SHA-256 mismatch: {relative}")
        records[relative] = dict(raw)
    required = {
        "homology_hits.tsv.gz",
        "identity_30/family_units.tsv",
        "identity_30/fold_manifest.tsv",
        "identity_30/split_summary.json",
        "source_manifest.json",
    }
    missing = sorted(required - set(records))
    if missing:
        raise FamilyExperimentError(f"family release is missing required files: {missing}")
    summary = json.loads((root / "identity_30" / "split_summary.json").read_text("utf-8"))
    validation = summary.get("validation")
    if not isinstance(validation, Mapping):
        raise FamilyExperimentError("family split validation block is missing")
    if int(validation.get("cross_fold_qualifying_edges", -1)) != 0:
        raise FamilyExperimentError("registered split has cross-fold qualifying edges")
    if int(validation.get("exact_duplicate_fold_violations", -1)) != 0:
        raise FamilyExperimentError("registered split separates exact duplicates")
    if not math.isclose(
        float(summary.get("minimum_bidirectional_coverage", -1.0)),
        MIN_BIDIRECTIONAL_COVERAGE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise FamilyExperimentError("family release coverage threshold changed")
    return {
        "verification_status": "verified",
        "manifest": _file_record(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "main_identity_percentage": MAIN_IDENTITY_PERCENT,
        "minimum_bidirectional_coverage": MIN_BIDIRECTIONAL_COVERAGE,
        "fold_manifest": records["identity_30/fold_manifest.tsv"],
        "split_summary": records["identity_30/split_summary.json"],
        "homology_hits": records["homology_hits.tsv.gz"],
        "cross_fold_qualifying_edges": 0,
        "exact_duplicate_fold_violations": 0,
    }


def load_family_assignments(
    path: Path, genes: Sequence[str]
) -> list[FamilyAssignment]:
    assignments: list[FamilyAssignment] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ["gene", "component_id", "fold", "stratum"]
        if reader.fieldnames != expected:
            raise FamilyExperimentError(
                f"family fold manifest columns changed: {reader.fieldnames!r}"
            )
        for row in reader:
            try:
                fold = int(row["fold"])
            except (TypeError, ValueError) as exc:
                raise FamilyExperimentError("non-integer family fold") from exc
            if fold not in range(N_FOLDS):
                raise FamilyExperimentError(f"family fold outside 0..4: {fold}")
            assignments.append(
                FamilyAssignment(
                    gene=row["gene"],
                    component_id=row["component_id"],
                    fold=fold,
                    stratum=row["stratum"],
                )
            )
    observed = [item.gene for item in assignments]
    if observed != list(genes):
        raise FamilyExperimentError("family manifest gene order does not match model genes")
    component_folds: dict[str, set[int]] = {}
    for item in assignments:
        component_folds.setdefault(item.component_id, set()).add(item.fold)
    split_components = [key for key, folds in component_folds.items() if len(folds) != 1]
    if split_components:
        raise FamilyExperimentError(
            f"sequence components span folds: {split_components[:5]}"
        )
    if set(item.fold for item in assignments) != set(range(N_FOLDS)):
        raise FamilyExperimentError("family manifest does not contain exactly five folds")
    return assignments


def fold_manifest_bytes(assignments: Sequence[FamilyAssignment]) -> bytes:
    """Serialize a fold assignment using the registered, byte-stable TSV layout."""
    lines = ["gene\tcomponent_id\tfold\tstratum"]
    lines.extend(
        f"{item.gene}\t{item.component_id}\t{item.fold}\t{item.stratum}"
        for item in assignments
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def derive_seeded_family_assignments(
    registered_manifest: Path,
    genes: Sequence[str],
    *,
    split_seed: int,
) -> list[FamilyAssignment]:
    """Reassign every complete registered sequence component for one split seed."""
    registered = load_family_assignments(registered_manifest, genes)
    members: dict[str, list[str]] = {}
    strata: dict[str, str] = {}
    for item in registered:
        members.setdefault(item.component_id, []).append(item.gene)
        previous = strata.setdefault(item.component_id, item.stratum)
        if previous != item.stratum:
            raise FamilyExperimentError(
                f"component {item.component_id} has inconsistent strata"
            )
    components = [
        (component_id, tuple(component_genes))
        for component_id, component_genes in members.items()
    ]
    fold_by_gene = assign_components_to_folds(
        components,
        fold_count=N_FOLDS,
        seed=int(split_seed),
    )
    assignments = [
        FamilyAssignment(
            gene=item.gene,
            component_id=item.component_id,
            fold=int(fold_by_gene[item.gene]),
            stratum=item.stratum,
        )
        for item in registered
    ]
    # Reuse the strict parser invariants by validating a complete in-memory view.
    if set(fold_by_gene) != set(genes):
        raise FamilyExperimentError("seeded family assignment is incomplete")
    component_folds: dict[str, set[int]] = {}
    for item in assignments:
        component_folds.setdefault(item.component_id, set()).add(item.fold)
    if any(len(folds) != 1 for folds in component_folds.values()):
        raise FamilyExperimentError("seeded assignment split a sequence component")
    if {item.fold for item in assignments} != set(range(N_FOLDS)):
        raise FamilyExperimentError("seeded assignment does not contain exactly five folds")
    payload = fold_manifest_bytes(assignments)
    if int(split_seed) == 42 and payload != registered_manifest.read_bytes():
        raise FamilyExperimentError(
            "split seed 42 did not reproduce the registered fold manifest byte-for-byte"
        )
    return assignments


def _materialize_fold_manifest(
    campaign_dir: Path,
    assignments: Sequence[FamilyAssignment],
    *,
    split_seed: int,
) -> dict[str, Any]:
    path = campaign_dir / "fold-manifest.tsv"
    payload = fold_manifest_bytes(assignments)
    if path.exists():
        if path.read_bytes() != payload:
            raise FamilyExperimentError("resumed fold manifest differs from seeded assignment")
    else:
        _atomic_write_bytes(path, payload)
    counts = [sum(item.fold == fold for item in assignments) for fold in range(N_FOLDS)]
    component_count = len({item.component_id for item in assignments})
    return {
        "schema": "DepPLM-family-holdout-fold-manifest-v2",
        "split_seed": int(split_seed),
        "assignment_algorithm": (
            "components_descending_size_then_sha256_seed_component_id; "
            "greedy_smallest_fold_then_lowest_fold_index"
        ),
        "gene_count": len(assignments),
        "component_count": component_count,
        "fold_gene_counts": counts,
        "seed_42_exact_registered_reproduction": (
            True if int(split_seed) == 42 else None
        ),
        "artifact": _file_record(path, relative_to=campaign_dir),
        "registered_component_source": _file_record(SPLIT_DIR / "fold_manifest.tsv"),
    }


def make_family_split(
    assignments: Sequence[FamilyAssignment], outer_fold: int
) -> FamilySplit:
    if outer_fold not in range(N_FOLDS):
        raise ValueError("outer_fold must be in 0..4")
    folds = np.asarray([item.fold for item in assignments], dtype=np.int64)
    inner_validation_fold = (outer_fold + 1) % N_FOLDS
    outer_test = np.flatnonzero(folds == outer_fold).astype(np.int64)
    outer_train = np.flatnonzero(folds != outer_fold).astype(np.int64)
    inner_validation = np.flatnonzero(folds == inner_validation_fold).astype(np.int64)
    inner_train = np.flatnonzero(
        (folds != outer_fold) & (folds != inner_validation_fold)
    ).astype(np.int64)
    sets = [set(inner_train), set(inner_validation), set(outer_test)]
    if any(sets[i].intersection(sets[j]) for i in range(3) for j in range(i + 1, 3)):
        raise FamilyExperimentError("nested family partitions overlap")
    if set(inner_train).union(inner_validation) != set(outer_train):
        raise FamilyExperimentError("inner gene partitions do not cover outer training genes")
    if set(outer_train).union(outer_test) != set(range(len(assignments))):
        raise FamilyExperimentError("outer family partitions do not cover all genes")
    return FamilySplit(
        outer_fold=outer_fold,
        inner_validation_fold=inner_validation_fold,
        outer_train=outer_train,
        outer_test=outer_test,
        inner_train=inner_train,
        inner_validation=inner_validation,
    )


def fit_training_gene_references(
    gene_effect: np.ndarray, train_genes: np.ndarray
) -> tuple[np.ndarray, float]:
    """Fit leakage-free baselines using only training-gene outcomes.

    A per-test-gene mean is undefined for a genuinely unseen gene.  The primary
    baseline is therefore one cell-specific intercept, fit by averaging the
    observed training genes in that cell.  The secondary baseline is a single
    global mean over the same training-gene matrix.
    """
    values = np.asarray(gene_effect, dtype=np.float32)
    indices = np.asarray(train_genes, dtype=np.int64)
    if values.ndim != 2 or indices.ndim != 1 or indices.size == 0:
        raise ValueError("gene_effect/train_genes have incompatible shapes")
    if np.unique(indices).size != indices.size:
        raise ValueError("train_genes contains duplicates")
    if int(indices.min()) < 0 or int(indices.max()) >= values.shape[0]:
        raise IndexError("train_genes refers outside gene_effect")
    train = values[indices]
    with np.errstate(invalid="ignore"):
        cell_mean = np.nanmean(train, axis=0, dtype=np.float64)
        global_mean = float(np.nanmean(train, dtype=np.float64))
    if not np.isfinite(cell_mean).all() or not np.isfinite(global_mean):
        raise FamilyExperimentError("training-gene baselines contain non-finite values")
    return cell_mean.astype(np.float32), global_mean


def fit_selectivity_labels(
    gene_effect: np.ndarray,
    train_genes: np.ndarray,
    *,
    top_frac: float = 0.25,
) -> np.ndarray:
    """Derive auxiliary labels exclusively for genes admitted to model fitting."""
    if not 0.0 < top_frac < 1.0:
        raise ValueError("top_frac must be between zero and one")
    values = np.asarray(gene_effect, dtype=np.float32)
    indices = np.asarray(train_genes, dtype=np.int64)
    train = values[indices]
    median = np.nanmedian(train, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(train - median), axis=1)
    finite = np.isfinite(mad)
    labels = np.zeros(indices.size, dtype=np.float32)
    if finite.any():
        threshold = np.nanquantile(mad[finite], 1.0 - top_frac)
        labels[finite] = (mad[finite] >= threshold).astype(np.float32)
    return labels


def fit_family_preprocessing(
    base: BaseData,
    *,
    seed: int,
    n_expr_pc: int,
    top_mut: int,
) -> FoldData:
    """Fit cell-covariate transforms; no gene-effect values are read here."""
    all_cells = np.arange(len(base.cell_ids), dtype=np.int64)
    expression_fit = fit_expression_pca(
        base.expression_raw,
        all_cells,
        n_components=n_expr_pc,
        seed=derive_seed(seed, "pca"),
    )
    expression = expression_fit.transform(base.expression_raw)
    mutation = fit_mutation_features(
        base.mutations_by_cell,
        base.genes,
        all_cells,
        top_n=top_mut,
        k_tokens=K_TOKENS,
    )
    lineage_fit = fit_lineage_one_hot(base.lineages, all_cells)
    lineage = lineage_fit.transform(base.lineages)
    edge_index_np = base.edge_index.T.astype(np.int64, copy=False)
    edge_weight_np = fit_comutation_edge_weights(
        edge_index_np,
        base.genes,
        mutation.matrix,
        mutation.vocabulary,
        all_cells,
    )
    edge_index = torch.from_numpy(edge_index_np.copy()).long().contiguous()
    edge_weight = torch.from_numpy(edge_weight_np.copy()).float()
    adjacency = normalized_adjacency(edge_index, edge_weight, len(base.genes))
    return FoldData(
        expression=expression,
        mutation=mutation,
        lineage=lineage,
        expression_fit=expression_fit,
        lineage_fit=lineage_fit,
        edge_index=edge_index,
        edge_weight=edge_weight,
        adjacency=adjacency,
        selectivity_labels=np.full(len(base.genes), np.nan, dtype=np.float32),
        fit_indices=all_cells,
    )


def preprocessing_record(fold_data: FoldData) -> dict[str, Any]:
    return {
        "fit_scope": "all_cells; gene_effect labels not accessed",
        "fit_cell_indices_sha256": _hash_array(fold_data.fit_indices.astype("<i8")),
        "n_cells": int(fold_data.fit_indices.size),
        "expression": {
            "input_features": int(fold_data.expression_fit.mean_.size),
            "output_components": int(fold_data.expression.shape[1]),
            "mean_sha256": _hash_array(fold_data.expression_fit.mean_),
            "scale_sha256": _hash_array(fold_data.expression_fit.scale_),
            "components_sha256": _hash_array(fold_data.expression_fit.components_),
        },
        "mutation": {
            "n_vocabulary": len(fold_data.mutation.vocabulary),
            "vocabulary_sha256": _hash_strings(fold_data.mutation.vocabulary),
            "token_order_sha256": _hash_strings(fold_data.mutation.token_order),
            "k_tokens": int(fold_data.mutation.tokens.shape[1]),
        },
        "lineage": {"categories": list(fold_data.lineage_fit.categories)},
        "graph": {
            "n_edges": int(fold_data.edge_index.shape[1]),
            "edge_weight_sha256": _hash_array(fold_data.edge_weight.numpy()),
            "n_non_neutral_weights": int(
                (~np.isclose(fold_data.edge_weight.numpy(), 1.0)).sum()
            ),
        },
    }


def _correlations(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    axis: int,
    rank: bool = False,
) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n_items = prediction.shape[0] if axis == 1 else prediction.shape[1]
    output = np.full(n_items, np.nan, dtype=np.float64)
    for index in range(n_items):
        x = prediction[index] if axis == 1 else prediction[:, index]
        y = target[index] if axis == 1 else target[:, index]
        finite = np.isfinite(x) & np.isfinite(y)
        if int(finite.sum()) < 3:
            continue
        x = x[finite]
        y = y[finite]
        if rank:
            x = rankdata(x)
            y = rankdata(y)
        if np.std(x) <= 0.0 or np.std(y) <= 0.0:
            continue
        output[index] = float(np.corrcoef(x, y)[0, 1])
    return output


def _finite_summary(
    values: np.ndarray, prefix: str
) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_sd": None,
            f"n_finite_{prefix}": 0,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_sd": float(np.std(finite)),
        f"n_finite_{prefix}": int(finite.size),
    }


def family_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    train_gene_cell_mean: np.ndarray,
    global_train_mean: float | np.ndarray,
    *,
    include_spearman: bool,
) -> dict[str, Any]:
    """Score unseen-gene predictions against training-gene-only references."""
    predicted = np.asarray(prediction, dtype=np.float32)
    observed = np.asarray(target, dtype=np.float32)
    if predicted.ndim != 2 or predicted.shape != observed.shape:
        raise ValueError("prediction and target must share a two-dimensional shape")
    reference = np.asarray(train_gene_cell_mean, dtype=np.float32)
    if reference.ndim == 1:
        if reference.size != predicted.shape[0]:
            raise ValueError("cell-mean reference must contain one value per cell")
        reference = np.broadcast_to(reference[:, None], predicted.shape)
    elif reference.shape != predicted.shape:
        raise ValueError("reference matrix has an incompatible shape")
    valid = np.isfinite(observed) & np.isfinite(reference)
    if not np.isfinite(predicted[valid]).all():
        raise ValueError("prediction contains non-finite evaluable values")
    n_pairs = int(valid.sum())
    if n_pairs == 0:
        raise ValueError("no finite evaluation pairs")
    error = np.where(valid, predicted - observed, 0.0)
    baseline_error = np.where(valid, reference - observed, 0.0)
    global_reference = np.asarray(global_train_mean, dtype=np.float32)
    if global_reference.ndim == 0:
        global_reference = np.broadcast_to(global_reference, predicted.shape)
        global_reference_value: float | None = float(global_train_mean)
        global_reference_scope = "single_outer_training_mean"
    elif global_reference.shape == predicted.shape:
        global_reference_value = None
        global_reference_scope = "fold_specific_outer_training_mean_matrix"
    else:
        raise ValueError("global training reference has an incompatible shape")
    if not np.isfinite(global_reference[valid]).all():
        raise ValueError("global training reference contains non-finite values")
    global_error = np.where(valid, global_reference - observed, 0.0)
    sse = float(np.square(error, dtype=np.float64).sum(dtype=np.float64))
    baseline_sse = float(
        np.square(baseline_error, dtype=np.float64).sum(dtype=np.float64)
    )
    global_sse = float(np.square(global_error, dtype=np.float64).sum(dtype=np.float64))
    absolute = float(np.abs(error).sum(dtype=np.float64))
    baseline_absolute = float(np.abs(baseline_error).sum(dtype=np.float64))
    result: dict[str, Any] = {
        "mse": sse / n_pairs,
        "rmse": math.sqrt(sse / n_pairs),
        "mae": absolute / n_pairs,
        "train_gene_set_cell_mean_mse": baseline_sse / n_pairs,
        "train_gene_set_cell_mean_mae": baseline_absolute / n_pairs,
        "residual_r2_vs_train_gene_set_cell_mean": (
            1.0 - sse / baseline_sse if baseline_sse > 0.0 else None
        ),
        "global_train_mean": global_reference_value,
        "global_train_reference_scope": global_reference_scope,
        "global_train_mean_mse": global_sse / n_pairs,
        "residual_r2_vs_global_train_mean": (
            1.0 - sse / global_sse if global_sse > 0.0 else None
        ),
        "n_finite_pairs": n_pairs,
        "n_cells": int(predicted.shape[0]),
        "n_held_out_genes": int(predicted.shape[1]),
    }
    raw_cell = _correlations(predicted, observed, axis=1)
    raw_gene = _correlations(predicted, observed, axis=0)
    residual_cell = _correlations(predicted - reference, observed - reference, axis=1)
    result.update(_finite_summary(raw_cell, "cellwise_pearson"))
    result.update(_finite_summary(raw_gene, "gene_wise_pearson"))
    result.update(_finite_summary(residual_cell, "residual_cellwise_pearson"))
    pooled_valid = valid.reshape(-1)
    flat_prediction = predicted.reshape(-1)[pooled_valid]
    flat_target = observed.reshape(-1)[pooled_valid]
    result["pooled_pearson"] = (
        float(np.corrcoef(flat_prediction, flat_target)[0, 1])
        if np.std(flat_prediction) > 0.0 and np.std(flat_target) > 0.0
        else None
    )
    if include_spearman:
        result.update(
            _finite_summary(
                _correlations(predicted, observed, axis=1, rank=True),
                "cellwise_spearman",
            )
        )
    return result


def _batch_inputs(
    fold_data: FoldData, cell_idx: np.ndarray, device: str
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(fold_data.mutation.matrix[cell_idx]).to(device),
        torch.from_numpy(fold_data.expression[cell_idx]).to(device),
        torch.from_numpy(fold_data.lineage[cell_idx]).to(device),
        torch.from_numpy(fold_data.mutation.tokens[cell_idx, :K_TOKENS]).to(device),
        torch.from_numpy(fold_data.mutation.token_mask[cell_idx, :K_TOKENS]).to(device),
    )


def predict_gene_subset(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    fold_data: FoldData,
    gene_indices: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    genes = torch.from_numpy(np.asarray(gene_indices, dtype=np.int64)).to(device)
    chunks: list[np.ndarray] = []
    all_cells = np.arange(fold_data.expression.shape[0], dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, all_cells.size, int(batch_size)):
            cell_idx = all_cells[start : start + int(batch_size)]
            inputs = _batch_inputs(fold_data, cell_idx, device)
            prediction, _ = _forward(model, config, *inputs, gene_subset=genes)
            chunks.append(prediction.float().cpu().numpy())
    output = np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)
    expected = (all_cells.size, len(gene_indices))
    if output.shape != expected or not np.isfinite(output).all():
        raise FamilyExperimentError(
            f"invalid prediction matrix {output.shape}; expected {expected}"
        )
    return output


def train_gene_model(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    base: BaseData,
    fold_data: FoldData,
    train_genes: np.ndarray,
    *,
    validation_genes: np.ndarray | None,
    epochs: int,
    scheduler_horizon: int,
    patience: int,
    batch_size: int,
    gene_subsample: int,
    device: str,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Train using only admitted gene labels; optionally score unseen validation genes."""
    train_genes = np.asarray(train_genes, dtype=np.int64)
    if train_genes.ndim != 1 or train_genes.size == 0:
        raise ValueError("train_genes must be a non-empty vector")
    if validation_genes is not None and set(train_genes).intersection(validation_genes):
        raise ValueError("training and validation genes overlap")
    set_global_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["lr"]),
        weight_decay=float(config["wd"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(scheduler_horizon)
    )
    rng = np.random.default_rng(derive_seed(seed, "family-batches"))
    selectivity_labels = fit_selectivity_labels(base.gene_effect, train_genes)
    selectivity_target = torch.from_numpy(selectivity_labels).to(device)
    selectivity_indices = torch.from_numpy(train_genes).to(device)
    validation_reference: tuple[np.ndarray, float] | None = None
    if validation_genes is not None:
        validation_reference = fit_training_gene_references(base.gene_effect, train_genes)

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_score = -np.inf
    best_metrics: dict[str, Any] | None = None
    bad_epochs = 0
    use_amp = device.startswith("cuda")
    all_cells = np.arange(len(base.cell_ids), dtype=np.int64)

    for epoch in range(int(epochs)):
        model.train()
        permutation = rng.permutation(all_cells)
        loss_sum = 0.0
        seen = 0
        for start in range(0, permutation.size, int(batch_size)):
            cell_idx = permutation[start : start + int(batch_size)]
            if 0 < gene_subsample < train_genes.size:
                selected = np.sort(
                    rng.choice(train_genes, int(gene_subsample), replace=False)
                ).astype(np.int64)
            else:
                selected = train_genes
            gene_tensor = torch.from_numpy(selected).to(device)
            target_np = np.asarray(
                base.gene_effect[np.ix_(selected, cell_idx)].T, dtype=np.float32
            )
            target = torch.from_numpy(target_np).to(device)
            inputs = _batch_inputs(fold_data, cell_idx, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                prediction, selectivity = _forward(
                    model, config, *inputs, gene_subset=gene_tensor
                )
                finite = torch.isfinite(target)
                residual = torch.where(
                    finite, prediction - target, torch.zeros_like(prediction)
                )
                loss = residual.square().sum() / finite.sum().clamp(min=1)
                if selectivity is not None:
                    selection_logits = selectivity[selectivity_indices]
                    loss = loss + float(config.get("lamb_sel", 0.0)) * (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            selection_logits, selectivity_target
                        )
                    )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * cell_idx.size
            seen += int(cell_idx.size)
        scheduler.step()

        epoch_record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(seen, 1),
        }
        if validation_genes is not None:
            if validation_reference is None:
                raise AssertionError("validation reference was not initialized")
            prediction = predict_gene_subset(
                model,
                config,
                fold_data,
                validation_genes,
                batch_size=batch_size,
                device=device,
            )
            target = np.asarray(base.gene_effect[validation_genes].T, dtype=np.float32)
            metrics = family_metrics(
                prediction,
                target,
                *validation_reference,
                include_spearman=False,
            )
            epoch_record["validation"] = metrics
            score = float(metrics["residual_r2_vs_train_gene_set_cell_mean"])
            if not np.isfinite(score):
                raise FamilyExperimentError("inner validation score is not finite")
            if score > best_score + 1e-8:
                best_score = score
                best_epoch = epoch + 1
                best_metrics = metrics
                bad_epochs = 0
            else:
                bad_epochs += 1
        history.append(epoch_record)
        message = (
            f"[family] epoch={epoch + 1}/{epochs} train_loss={epoch_record['train_loss']:.6f}"
        )
        if validation_genes is not None:
            message += f" val_residual_r2={score:.6f}"
        print(message, flush=True)
        if validation_genes is not None and bad_epochs >= int(patience):
            break

    record: dict[str, Any] = {
        "trained_epochs": len(history),
        "scheduler": "CosineAnnealingLR",
        "scheduler_horizon_epochs": int(scheduler_horizon),
        "training_gene_indices_sha256": _hash_array(train_genes.astype("<i8")),
        "n_training_genes": int(train_genes.size),
        "selectivity_labels_scope": "training_genes_only",
        "selectivity_labels_sha256": _hash_array(selectivity_labels),
        "history": history,
    }
    if validation_genes is not None:
        if best_metrics is None or best_epoch < 1:
            raise FamilyExperimentError("no valid inner-validation checkpoint")
        record.update(
            {
                "selection_metric": "residual_r2_vs_train_gene_set_cell_mean",
                "selection_score": float(best_score),
                "best_epoch": int(best_epoch),
                "best_validation_metrics": best_metrics,
            }
        )
    return model, record


def _split_record(
    split: FamilySplit,
    assignments: Sequence[FamilyAssignment],
    base: BaseData,
) -> dict[str, Any]:
    def block(indices: np.ndarray) -> dict[str, Any]:
        components = sorted({assignments[int(index)].component_id for index in indices})
        return {
            "n_genes": int(indices.size),
            "n_components": len(components),
            "gene_indices": [int(value) for value in indices],
            "source_gene_indices": [
                int(base.source_gene_indices[int(value)]) for value in indices
            ],
            "gene_indices_sha256": _hash_array(indices.astype("<i8")),
            "gene_ids_sha256": _hash_strings([base.genes[int(value)] for value in indices]),
            "component_ids_sha256": _hash_strings(components),
        }

    return {
        "schema": "DepPLM-family-holdout-nested-split-v1",
        "holdout_axis": "genes",
        "cell_holdout": False,
        "outer_fold": split.outer_fold,
        "inner_validation_fold": split.inner_validation_fold,
        "main_identity_percentage": MAIN_IDENTITY_PERCENT,
        "minimum_bidirectional_coverage": MIN_BIDIRECTIONAL_COVERAGE,
        "outer_train": block(split.outer_train),
        "inner_train": block(split.inner_train),
        "inner_validation": block(split.inner_validation),
        "outer_test": block(split.outer_test),
        "n_evaluation_cells": len(base.cell_ids),
        "evaluation_cell_ids_sha256": _hash_strings(base.cell_ids),
        "label_isolation": (
            "outer-test gene-effect labels are accessed only by the one final "
            "evaluation after tuning and fresh outer-train refit"
        ),
    }


def _prediction_arrays(
    prediction: np.ndarray,
    train_gene_cell_mean: np.ndarray,
    global_train_mean: float,
    split: FamilySplit,
    base: BaseData,
) -> dict[str, np.ndarray]:
    return {
        "prediction": np.asarray(prediction, dtype=np.float32),
        "train_gene_set_cell_mean": np.asarray(train_gene_cell_mean, dtype=np.float32),
        "global_train_mean": np.asarray([global_train_mean], dtype=np.float32),
        "outer_test_gene_indices": split.outer_test.astype(np.int64),
        "outer_train_gene_indices": split.outer_train.astype(np.int64),
        "inner_train_gene_indices": split.inner_train.astype(np.int64),
        "inner_validation_gene_indices": split.inner_validation.astype(np.int64),
        "source_test_gene_indices": base.source_gene_indices[split.outer_test].astype(
            np.int64
        ),
        "source_cell_indices": base.source_cell_indices.astype(np.int64),
        "test_gene_ids": np.asarray(
            [base.genes[int(value)] for value in split.outer_test], dtype=str
        ),
        "cell_ids": np.asarray(base.cell_ids, dtype=str),
    }


def _publish_condition(
    final_dir: Path,
    *,
    prediction: np.ndarray,
    references: tuple[np.ndarray, float],
    split: FamilySplit,
    base: BaseData,
    result: dict[str, Any],
) -> dict[str, Any]:
    if final_dir.exists():
        return validate_condition_artifact(final_dir, base, split)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = final_dir.parent / f".{final_dir.name}.partial-{uuid.uuid4().hex}"
    stage.mkdir(exist_ok=False)
    try:
        prediction_record = _atomic_write_npz(
            stage / "outer-test-predictions.npz",
            _prediction_arrays(prediction, *references, split, base),
        )
        target = np.asarray(base.gene_effect[split.outer_test].T, dtype=np.float32)
        result = {
            **result,
            "outer_test_predictions": prediction_record,
            "outer_test_target": {
                "source": "data/processed/gene_effect.npy",
                "shape": list(target.shape),
                "dtype": str(target.dtype),
                "sha256": _hash_array(target),
                "persisted_in_prediction_npz": False,
            },
        }
        _atomic_write_json(stage / "result.json", result)
        os.replace(stage, final_dir)
    except OSError:
        if final_dir.exists():
            raise AtomicWriteError(f"refusing to overwrite condition directory: {final_dir}")
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return validate_condition_artifact(final_dir, base, split)


def validate_condition_artifact(
    condition_dir: Path,
    base: BaseData,
    split: FamilySplit,
) -> dict[str, Any]:
    result_path = condition_dir / "result.json"
    prediction_path = condition_dir / "outer-test-predictions.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != SCHEMA:
        raise FamilyExperimentError("condition result schema changed")
    if result.get("condition") != condition_dir.name:
        raise FamilyExperimentError("condition result is stored under the wrong directory")
    if int(result.get("outer_fold", -1)) != split.outer_fold:
        raise FamilyExperimentError("condition result is stored under the wrong fold")
    if result.get("holdout_axis") != "genes" or result.get("cell_holdout") is not False:
        raise FamilyExperimentError("condition result misstates the holdout axis")
    expected_hash = result["outer_test_predictions"]["sha256"]
    if _sha256_file(prediction_path) != expected_hash:
        raise FamilyExperimentError(f"prediction hash mismatch: {prediction_path}")
    with np.load(prediction_path, allow_pickle=False) as archive:
        expected_names = set(result["outer_test_predictions"]["arrays"])
        if set(archive.files) != expected_names:
            raise FamilyExperimentError("prediction array names disagree with result")
        for name in archive.files:
            value = archive[name]
            record = result["outer_test_predictions"]["arrays"][name]
            if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
                raise FamilyExperimentError(f"prediction array metadata mismatch: {name}")
            if _hash_array(value) != record["sha256"]:
                raise FamilyExperimentError(f"prediction array hash mismatch: {name}")
        if not np.array_equal(archive["outer_test_gene_indices"], split.outer_test):
            raise FamilyExperimentError("prediction artifact has wrong outer-test genes")
        for name, expected in (
            ("outer_train_gene_indices", split.outer_train),
            ("inner_train_gene_indices", split.inner_train),
            ("inner_validation_gene_indices", split.inner_validation),
        ):
            if not np.array_equal(archive[name], expected):
                raise FamilyExperimentError(f"prediction artifact has wrong {name}")
        if archive["test_gene_ids"].tolist() != [
            base.genes[int(index)] for index in split.outer_test
        ]:
            raise FamilyExperimentError("prediction artifact has wrong test gene IDs")
        if archive["cell_ids"].tolist() != list(base.cell_ids):
            raise FamilyExperimentError("prediction artifact has wrong cell IDs")
        if not np.array_equal(
            archive["source_test_gene_indices"],
            base.source_gene_indices[split.outer_test],
        ):
            raise FamilyExperimentError("prediction artifact has wrong source gene indices")
        if not np.array_equal(archive["source_cell_indices"], base.source_cell_indices):
            raise FamilyExperimentError("prediction artifact has wrong source cell indices")
        prediction = np.asarray(archive["prediction"], dtype=np.float32)
        if prediction.shape != (len(base.cell_ids), split.outer_test.size):
            raise FamilyExperimentError("prediction artifact has wrong prediction shape")
        if not np.isfinite(prediction).all():
            raise FamilyExperimentError("prediction artifact contains non-finite predictions")
        target = np.asarray(base.gene_effect[split.outer_test].T, dtype=np.float32)
        if _hash_array(target) != result["outer_test_target"]["sha256"]:
            raise FamilyExperimentError("outer-test target binding changed")
    return result


def _run_baseline_fold(
    fold_dir: Path,
    split: FamilySplit,
    base: BaseData,
    split_record: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    references = fit_training_gene_references(base.gene_effect, split.outer_train)
    cell_mean, global_mean = references
    prediction = np.broadcast_to(
        cell_mean[:, None], (len(base.cell_ids), split.outer_test.size)
    ).copy()
    target = np.asarray(base.gene_effect[split.outer_test].T, dtype=np.float32)
    metrics = family_metrics(
        prediction, target, cell_mean, global_mean, include_spearman=True
    )
    return _publish_condition(
        fold_dir / BASELINE_CONDITION,
        prediction=prediction,
        references=references,
        split=split,
        base=base,
        result={
            "schema": SCHEMA,
            "condition": BASELINE_CONDITION,
            "outer_fold": split.outer_fold,
            "holdout_axis": "genes",
            "cell_holdout": False,
            "split_seed": int(args.split_seed),
            "training_seed": int(args.training_seed),
            "baseline_definition": (
                "For each observed cell, predict every unseen test gene by the mean "
                "gene-effect across outer-training genes in that same cell. No outer-test "
                "gene label is used. A per-test-gene mean is undefined for unseen genes."
            ),
            "split_sha256": _canonical_json_sha256(dict(split_record)),
            "outer_test_evaluation_calls": 1,
            "test_metrics": metrics,
            "scientific_result_scope": "single_outer_family_fold",
        },
    )


def _run_neural_fold(
    condition: str,
    fold_dir: Path,
    split: FamilySplit,
    base: BaseData,
    fold_data: FoldData,
    split_record: Mapping[str, Any],
    grid: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    final_dir = fold_dir / condition
    if final_dir.exists():
        return validate_condition_artifact(final_dir, base, split)
    started = time.time()
    fold_seed = derive_seed(
        args.training_seed,
        "family-training",
        split.outer_fold,
        condition,
    )
    candidates = [{**NEURAL_CONDITIONS[condition], **candidate} for candidate in grid]
    tuning: list[dict[str, Any]] = []
    for config_index, config in enumerate(candidates):
        config_seed = derive_seed(fold_seed, "config", config_index)
        print(
            f"[family] fold={split.outer_fold} condition={condition} "
            f"config={config_index + 1}/{len(candidates)} {config}",
            flush=True,
        )
        set_global_seed(config_seed)
        model = build_model(config, base, fold_data)
        model, training = train_gene_model(
            model,
            config,
            base,
            fold_data,
            split.inner_train,
            validation_genes=split.inner_validation,
            epochs=args.inner_max_epochs,
            scheduler_horizon=args.inner_max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            gene_subsample=args.gene_subsample,
            device=device,
            seed=config_seed,
        )
        tuning.append(
            {
                "config_index": config_index,
                "config": dict(config),
                "seed": int(config_seed),
                "selection_score": float(training["selection_score"]),
                "best_epoch": int(training["best_epoch"]),
                "training": training,
            }
        )
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    selected = max(
        tuning,
        key=lambda item: (float(item["selection_score"]), -int(item["config_index"])),
    )
    config = dict(selected["config"])
    best_epoch = int(selected["best_epoch"])
    refit_seed = derive_seed(fold_seed, "outer-refit")
    print(
        f"[family] fold={split.outer_fold} condition={condition} "
        f"refit_epochs={best_epoch} config={config}",
        flush=True,
    )
    set_global_seed(refit_seed)
    model = build_model(config, base, fold_data)
    model, refit = train_gene_model(
        model,
        config,
        base,
        fold_data,
        split.outer_train,
        validation_genes=None,
        epochs=best_epoch,
        scheduler_horizon=args.inner_max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        gene_subsample=args.gene_subsample,
        device=device,
        seed=refit_seed,
    )
    references = fit_training_gene_references(base.gene_effect, split.outer_train)
    prediction = predict_gene_subset(
        model,
        config,
        fold_data,
        split.outer_test,
        batch_size=args.batch_size,
        device=device,
    )
    target = np.asarray(base.gene_effect[split.outer_test].T, dtype=np.float32)
    metrics = family_metrics(
        prediction, target, *references, include_spearman=True
    )
    result = _publish_condition(
        final_dir,
        prediction=prediction,
        references=references,
        split=split,
        base=base,
        result={
            "schema": SCHEMA,
            "condition": condition,
            "outer_fold": split.outer_fold,
            "holdout_axis": "genes",
            "cell_holdout": False,
            "split_seed": int(args.split_seed),
            "training_seed": int(args.training_seed),
            "feature_access": (
                "label-inductive, feature-transductive: frozen sequence embeddings, "
                "the fixed PPI graph, and cell covariates are available for all genes; "
                "outer-test gene-effect labels are never used for fitting or selection"
            ),
            "split_sha256": _canonical_json_sha256(dict(split_record)),
            "selected_config": config,
            "selected_config_index": int(selected["config_index"]),
            "selected_epoch": best_epoch,
            "selection_metric": "residual_r2_vs_train_gene_set_cell_mean",
            "selection_score": float(selected["selection_score"]),
            "tuning": tuning,
            "refit_seed": int(refit_seed),
            "refit": refit,
            "outer_test_evaluation_calls": 1,
            "test_metrics": metrics,
            "wall_seconds": time.time() - started,
            "scientific_result_scope": "single_outer_family_fold",
        },
    )
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _installed_distributions() -> list[dict[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or distribution.metadata.get("Summary")
        if not name:
            continue
        records[name.casefold()] = (str(name), str(distribution.version))
    return [
        {"name": name, "version": version}
        for _, (name, version) in sorted(records.items())
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
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _source_paths() -> list[Path]:
    return [
        Path(__file__).resolve(),
        ROOT / "code" / "family_holdout.py",
        ROOT / "code" / "verify_family_holdout_experiment.py",
        ROOT / "code" / "run_corrected_experiments.py",
        ROOT / "code" / "corrected_protocol.py",
        ROOT / "code" / "models.py",
        ROOT / "code" / "03_embed_proteins.py",
        ROOT / "code" / "embedding_provenance.py",
        PROCESSED / "gene_effect.npy",
        ROOT / EMBEDDING_RELATIVE_PATH,
        ROOT / NORMALIZED_RELATIVE_PATH,
        ROOT / NAMES_RELATIVE_PATH,
        ROOT / MANIFEST_RELATIVE_PATH,
        PROCESSED / "gene_seqs.fa",
        PROCESSED / "ppi_graph.npz",
        PROCESSED / "genes.txt",
        PROCESSED / "cell_lines.txt",
        PROCESSED / "lineages.txt",
        DEPMAP / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        DEPMAP / "OmicsSomaticMutations.csv",
        FAMILY_ROOT / "release_manifest.json",
        SPLIT_DIR / "fold_manifest.tsv",
        SPLIT_DIR / "split_summary.json",
    ]


def _validate_source_stability(records: Sequence[Mapping[str, Any]]) -> None:
    current = [_file_record(path) for path in _source_paths()]
    if len(records) != len(current):
        raise FamilyExperimentError("source inventory length changed during run")
    for before, after in zip(records, current):
        if any(before.get(key) != after.get(key) for key in ("path", "bytes", "sha256")):
            raise FamilyExperimentError(
                f"source changed during family campaign: {before.get('path')}"
            )


def _create_campaign_dir(
    output_root: Path,
    *,
    split_seed: int,
    training_seed: int,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = output_root / (
        f"split-{split_seed}-training-{training_seed}-"
        f"{timestamp}-{uuid.uuid4().hex[:8]}"
    )
    target.mkdir(exist_ok=False)
    return target


def _write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create a checkpoint descriptor, or fail if a resumed one differs."""
    expected = _json_safe(dict(value))
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise FamilyExperimentError(f"resumed checkpoint metadata changed: {path}")
        return
    _atomic_write_json(path, expected)


def _write_or_validate_completion(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Resume safely across the completion-to-release publication boundary."""
    expected = _json_safe(dict(value))
    if not path.exists():
        _atomic_write_json(path, expected)
        return expected
    observed = json.loads(path.read_text(encoding="utf-8"))
    observed_stable = {key: item for key, item in observed.items() if key != "completed_utc"}
    expected_stable = {key: item for key, item in expected.items() if key != "completed_utc"}
    if observed_stable != expected_stable:
        raise FamilyExperimentError("resumed completion metadata changed")
    return observed


def _campaign_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "split_seed": int(args.split_seed),
        "training_seed": int(args.training_seed),
        "fold_id": int(args.fold_id),
        "conditions": list(args.conditions),
        "inner_max_epochs": int(args.inner_max_epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "gene_subsample": int(args.gene_subsample),
        "top_mut": int(args.top_mut),
        "n_expr_pc": int(args.n_expr_pc),
        "pilot": bool(args.pilot),
        "pilot_cells": int(args.pilot_cells),
        "pilot_genes": int(args.pilot_genes),
    }


def _canonical_reasons(args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if args.pilot:
        reasons.append("pilot mode is a structural smoke test")
    if args.split_seed not in REGISTERED_SPLIT_SEEDS:
        reasons.append(
            f"split_seed {args.split_seed} is not in {REGISTERED_SPLIT_SEEDS}"
        )
    if args.training_seed not in REGISTERED_TRAINING_SEEDS:
        reasons.append(
            f"training_seed {args.training_seed} is not in {REGISTERED_TRAINING_SEEDS}"
        )
    if args.fold_id != -1:
        reasons.append("a single --fold-id is a partial campaign")
    if tuple(args.conditions) != tuple(NEURAL_CONDITIONS):
        reasons.append("canonical campaign requires DepPLM and C-PLM-MLP")
    for key, expected in CANONICAL_ARGUMENTS.items():
        if int(getattr(args, key)) != expected:
            reasons.append(f"non-canonical {key}={getattr(args, key)}; expected {expected}")
    return reasons


def _initialize_campaign(
    campaign_dir: Path,
    args: argparse.Namespace,
    *,
    family_provenance: Mapping[str, Any],
    embedding_provenance: Mapping[str, Any],
    fold_manifest: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    device: str,
) -> None:
    distributions = _installed_distributions()
    cuda_available = bool(torch.cuda.is_available())
    payload = {
        "schema": SCHEMA,
        "status": "initialized",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "holdout_axis": "genes",
            "cell_holdout": False,
            "outer_folds": N_FOLDS,
            "inner_validation_rule": "(outer_fold + 1) mod 5",
            "main_identity_percentage": MAIN_IDENTITY_PERCENT,
            "minimum_bidirectional_coverage": MIN_BIDIRECTIONAL_COVERAGE,
            "feature_access": "label-inductive, feature-transductive",
            "outer_test_access": "one call after selection and fresh refit",
        },
        "configuration": _campaign_config(args),
        "canonical_deviations": _canonical_reasons(args),
        "device": device,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if device == "cuda" else None,
            "cuda_available": cuda_available,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if device == "cuda" else None
            ),
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "installed_distributions": distributions,
            "installed_distributions_sha256": _canonical_json_sha256(
                {"installed_distributions": distributions}
            ),
        },
        "git": _git_record(),
        "family_release": dict(family_provenance),
        "fold_manifest": dict(fold_manifest),
        "esm_embedding": dict(embedding_provenance),
        "sources": list(source_records),
    }
    _atomic_write_json(campaign_dir / "run-manifest.json", payload)


def _validate_resume_campaign(
    campaign_dir: Path,
    args: argparse.Namespace,
    source_records: Sequence[Mapping[str, Any]],
    *,
    fold_manifest: Mapping[str, Any],
    device: str,
) -> None:
    manifest_path = campaign_dir / "run-manifest.json"
    if not manifest_path.is_file():
        raise FamilyExperimentError("resume directory has no run-manifest.json")
    if (campaign_dir / "release-manifest.json").exists():
        raise FamilyExperimentError("refusing to resume an already finalized campaign")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise FamilyExperimentError("resume campaign schema changed")
    if manifest.get("configuration") != _campaign_config(args):
        raise FamilyExperimentError("resume arguments differ from initialized campaign")
    if manifest.get("device") != device:
        raise FamilyExperimentError("resume device differs from initialized campaign")
    if manifest.get("sources") != list(source_records):
        raise FamilyExperimentError("resume source inventory differs from initialized campaign")
    if manifest.get("fold_manifest") != dict(fold_manifest):
        raise FamilyExperimentError("resume fold manifest differs from initialized campaign")
    distributions = _installed_distributions()
    current_environment_hash = _canonical_json_sha256(
        {"installed_distributions": distributions}
    )
    if (
        manifest.get("environment", {}).get("installed_distributions_sha256")
        != current_environment_hash
    ):
        raise FamilyExperimentError("installed environment changed since initialization")


def _load_prediction(
    campaign_dir: Path,
    fold: int,
    condition: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    condition_dir = campaign_dir / f"fold-{fold:02d}" / condition
    result = json.loads((condition_dir / "result.json").read_text(encoding="utf-8"))
    prediction_path = condition_dir / "outer-test-predictions.npz"
    if _sha256_file(prediction_path) != result["outer_test_predictions"]["sha256"]:
        raise FamilyExperimentError(f"aggregate prediction hash mismatch: {prediction_path}")
    with np.load(prediction_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return result, arrays


def _fold_descriptive_summary(values: Sequence[float | None]) -> dict[str, Any]:
    safe_values = [
        None if value is None or not math.isfinite(float(value)) else float(value)
        for value in values
    ]
    finite = [value for value in safe_values if value is not None]
    if len(finite) != len(safe_values) or len(finite) < 2:
        return {
            "status": "undefined_nonfinite_fold_metric",
            "n_folds": len(safe_values),
            "n_finite_folds": len(finite),
            "mean": None,
            "sd": None,
            "minimum": None,
            "maximum": None,
            "fold_values": safe_values,
        }
    array = np.asarray(finite, dtype=np.float64)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    return {
        "status": "descriptive_fold_dispersion",
        "n_folds": int(array.size),
        "n_finite_folds": int(array.size),
        "mean": mean,
        "sd": sd,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "fold_values": array.tolist(),
    }


def aggregate_campaign(
    campaign_dir: Path,
    base: BaseData,
    assignments: Sequence[FamilyAssignment],
    folds: Sequence[int],
    conditions: Sequence[str],
) -> dict[str, Any]:
    expected_folds = list(range(N_FOLDS))
    if list(folds) != expected_folds:
        raise FamilyExperimentError("scientific aggregation requires all five outer folds")
    all_conditions = [*conditions, BASELINE_CONDITION]
    condition_summaries: dict[str, Any] = {}
    per_fold_metrics: dict[str, list[dict[str, Any]]] = {}
    for condition in all_conditions:
        full_prediction = np.full(
            (len(base.cell_ids), len(base.genes)), np.nan, dtype=np.float32
        )
        full_reference = np.full_like(full_prediction, np.nan)
        full_global_reference = np.full_like(full_prediction, np.nan)
        fold_metrics: list[dict[str, Any]] = []
        prediction_records: list[dict[str, Any]] = []
        for fold in expected_folds:
            split = make_family_split(assignments, fold)
            result, arrays = _load_prediction(campaign_dir, fold, condition)
            if not np.array_equal(arrays["outer_test_gene_indices"], split.outer_test):
                raise FamilyExperimentError("aggregate fold genes do not match split")
            prediction = np.asarray(arrays["prediction"], dtype=np.float32)
            expected_shape = (len(base.cell_ids), split.outer_test.size)
            if prediction.shape != expected_shape:
                raise FamilyExperimentError("aggregate prediction has wrong shape")
            if np.isfinite(full_prediction[:, split.outer_test]).any():
                raise FamilyExperimentError("a gene received predictions from multiple folds")
            full_prediction[:, split.outer_test] = prediction
            cell_mean = np.asarray(arrays["train_gene_set_cell_mean"], dtype=np.float32)
            full_reference[:, split.outer_test] = cell_mean[:, None]
            full_global_reference[:, split.outer_test] = float(
                arrays["global_train_mean"][0]
            )
            fold_metrics.append(dict(result["test_metrics"]))
            prediction_records.append(
                {
                    "outer_fold": fold,
                    "path": str(
                        (campaign_dir / f"fold-{fold:02d}" / condition / "outer-test-predictions.npz")
                        .relative_to(campaign_dir)
                        .as_posix()
                    ),
                    "sha256": result["outer_test_predictions"]["sha256"],
                }
            )
        if (
            not np.isfinite(full_prediction).all()
            or not np.isfinite(full_reference).all()
            or not np.isfinite(full_global_reference).all()
        ):
            raise FamilyExperimentError("aggregate predictions do not cover every gene exactly once")
        target = np.asarray(base.gene_effect.T, dtype=np.float32)
        pooled = family_metrics(
            full_prediction,
            target,
            full_reference,
            full_global_reference,
            include_spearman=True,
        )
        condition_summaries[condition] = {
            "pooled_outer_predictions": pooled,
            "fold_mean_sd": {
                metric: _fold_descriptive_summary(
                    [record[metric] for record in fold_metrics]
                )
                for metric in (
                    "mse",
                    "mae",
                    "residual_r2_vs_train_gene_set_cell_mean",
                    "cellwise_pearson_mean",
                    "gene_wise_pearson_mean",
                    "cellwise_spearman_mean",
                )
            },
            "prediction_artifacts": prediction_records,
        }
        per_fold_metrics[condition] = fold_metrics

    comparisons: dict[str, Any] = {}
    for comparator in ("C-PLM-MLP", BASELINE_CONDITION):
        comparisons[f"DepPLM_minus_{comparator}"] = {
            metric: _fold_descriptive_summary(
                [
                    (
                        float(per_fold_metrics["DepPLM"][fold][metric])
                        - float(per_fold_metrics[comparator][fold][metric])
                        if per_fold_metrics["DepPLM"][fold][metric] is not None
                        and per_fold_metrics[comparator][fold][metric] is not None
                        else None
                    )
                    for fold in expected_folds
                ]
            )
            for metric in (
                "mse",
                "mae",
                "residual_r2_vs_train_gene_set_cell_mean",
                "cellwise_pearson_mean",
                "gene_wise_pearson_mean",
                "cellwise_spearman_mean",
            )
        }
    return {
        "schema": "DepPLM-family-holdout-aggregate-v2",
        "holdout_axis": "genes",
        "cell_holdout": False,
        "outer_folds": expected_folds,
        "conditions": condition_summaries,
        "paired_fold_differences": comparisons,
        "inference_note": (
            "Fold summaries are descriptive dispersion checks only. Folds and fit "
            "seeds are not independent biological replicates; repeat-aware paired "
            "summaries are produced only by the registered top-level campaign."
        ),
        "scientific_result_scope": "one_split_seed_by_one_training_seed_cell",
        "top_level_campaign_aggregation_required": True,
    }


def _release_manifest(campaign_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(campaign_dir.rglob("*")):
        relative = path.relative_to(campaign_dir)
        if (
            not path.is_file()
            or path.name == "release-manifest.json"
            or any(part.startswith(".") for part in relative.parts)
        ):
            continue
        files.append(_file_record(path, relative_to=campaign_dir))
    return {
        "schema": "DepPLM-family-holdout-model-release-v2",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }


def _device_from_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return value


def _at_least(minimum: int):
    def convert(value: str) -> int:
        parsed = int(value)
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return parsed

    return convert


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-seed",
        type=int,
        choices=REGISTERED_SPLIT_SEEDS,
        default=REGISTERED_SPLIT_SEEDS[0],
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        choices=REGISTERED_TRAINING_SEEDS,
        default=REGISTERED_TRAINING_SEEDS[0],
    )
    parser.add_argument("--fold-id", type=int, choices=[-1, 0, 1, 2, 3, 4], default=-1)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(NEURAL_CONDITIONS),
        default=list(NEURAL_CONDITIONS),
    )
    parser.add_argument("--inner-max-epochs", type=_at_least(1), default=20)
    parser.add_argument("--patience", type=_at_least(1), default=5)
    parser.add_argument("--batch-size", type=_at_least(1), default=64)
    parser.add_argument("--gene-subsample", type=_at_least(0), default=6_000)
    parser.add_argument("--top-mut", type=_at_least(1), default=5_000)
    parser.add_argument("--n-expr-pc", type=_at_least(1), default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-cells", type=_at_least(10), default=64)
    parser.add_argument("--pilot-genes", type=_at_least(25), default=250)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        help=(
            "Explicit create-or-resume directory. An existing incomplete campaign is "
            "resumed only when arguments, source hashes, device, and environment match."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("--conditions must not contain duplicates")
    if args.pilot:
        args.inner_max_epochs = 1
        args.patience = 1
        args.batch_size = min(args.batch_size, 32)
        args.gene_subsample = 0
        args.top_mut = min(args.top_mut, 128)
        args.n_expr_pc = min(args.n_expr_pc, 32)
    device = _device_from_arg(args.device)
    set_global_seed(args.training_seed)
    family_provenance = validate_family_release()
    embedding_provenance = validate_embedding_bundle(ROOT)
    source_records = [_file_record(path) for path in _source_paths()]
    base = load_base_data(
        pilot=args.pilot,
        seed=args.training_seed,
        pilot_cells=args.pilot_cells,
        pilot_genes=args.pilot_genes,
    )
    full_genes = [
        line.strip()
        for line in (PROCESSED / "genes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    full_assignments = derive_seeded_family_assignments(
        SPLIT_DIR / "fold_manifest.tsv",
        full_genes,
        split_seed=args.split_seed,
    )
    assignment_by_gene = {item.gene: item for item in full_assignments}
    assignments = [assignment_by_gene[gene] for gene in base.genes]
    if args.campaign_dir is None:
        campaign_dir = _create_campaign_dir(
            args.output_root.resolve(),
            split_seed=args.split_seed,
            training_seed=args.training_seed,
        )
        resumed = False
    else:
        campaign_dir = args.campaign_dir.resolve()
        resumed = campaign_dir.exists()
        if not resumed:
            campaign_dir.mkdir(parents=True, exist_ok=False)
    fold_manifest = _materialize_fold_manifest(
        campaign_dir,
        full_assignments,
        split_seed=args.split_seed,
    )
    if resumed:
        _validate_resume_campaign(
            campaign_dir,
            args,
            source_records,
            fold_manifest=fold_manifest,
            device=device,
        )
    else:
        _initialize_campaign(
            campaign_dir,
            args,
            family_provenance=family_provenance,
            embedding_provenance=embedding_provenance,
            fold_manifest=fold_manifest,
            source_records=source_records,
            device=device,
        )
    print(f"[family] campaign={campaign_dir} resumed={resumed}", flush=True)
    print("[family] fitting cell-covariate preprocessing once", flush=True)
    fold_data = fit_family_preprocessing(
        base,
        seed=args.training_seed,
        n_expr_pc=args.n_expr_pc,
        top_mut=args.top_mut,
    )
    _write_or_validate_json(
        campaign_dir / "preprocessing.json", preprocessing_record(fold_data)
    )
    grid = FULL_GRID[:1] if args.pilot else FULL_GRID
    folds = list(range(N_FOLDS)) if args.fold_id == -1 else [args.fold_id]
    for fold in folds:
        split = make_family_split(assignments, fold)
        fold_dir = campaign_dir / f"fold-{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        split_record = _split_record(split, assignments, base)
        _write_or_validate_json(fold_dir / "split.json", split_record)
        print(f"[family] fold={fold}: training-only baseline", flush=True)
        _run_baseline_fold(fold_dir, split, base, split_record, args)
        for condition in args.conditions:
            _run_neural_fold(
                condition,
                fold_dir,
                split,
                base,
                fold_data,
                split_record,
                grid,
                args,
                device,
            )
    _validate_source_stability(source_records)
    deviations = _canonical_reasons(args)
    if (
        folds == list(range(N_FOLDS))
        and not args.pilot
        and tuple(args.conditions) == tuple(NEURAL_CONDITIONS)
    ):
        aggregate = aggregate_campaign(
            campaign_dir, base, assignments, folds, args.conditions
        )
        _write_or_validate_json(campaign_dir / "aggregate-summary.json", aggregate)
    completion = {
        "schema": SCHEMA,
        "status": "campaign_cell_complete" if not deviations else "partial_or_noncanonical",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_result": False,
        "campaign_aggregation_required": True,
        "registered_campaign_cell_eligible": not deviations,
        "canonical_deviations": deviations,
        "completed_folds": folds,
        "completed_conditions": [*args.conditions, BASELINE_CONDITION],
        "source_stability": "verified",
        "holdout_axis": "genes",
        "cell_holdout": False,
    }
    completion = _write_or_validate_completion(
        campaign_dir / "completion.json", completion
    )
    _atomic_write_json(campaign_dir / "release-manifest.json", _release_manifest(campaign_dir))
    print(
        f"[family] completed status={completion['status']} campaign={campaign_dir}",
        flush=True,
    )
    return campaign_dir


if __name__ == "__main__":
    main()
