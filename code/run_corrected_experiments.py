"""Run leakage-resistant DepPLM P1/P2 evaluations.

This entry point intentionally does not import training helpers from the legacy
runner.  Every learned preprocessing artifact is fit inside the relevant
training split, hyperparameters and epoch count are selected on an inner
validation split, and the outer test split is evaluated exactly once after a
fresh fixed-epoch refit.

Pilot mode is a structural smoke test only.  It uses fewer cells, genes,
expression columns, configurations and epochs and must not be reported as a
scientific result.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from corrected_protocol import (
    AtomicWriteError,
    K_TOKENS,
    FittedExpressionPCA,
    FittedLineageEncoder,
    MutationFeatures,
    atomic_write_json,
    build_split_manifest,
    dependency_metrics,
    derive_seed,
    fit_comutation_edge_weights,
    fit_expression_pca,
    fit_gene_mean_baseline,
    fit_lineage_one_hot,
    fit_mutation_features,
    fit_validation_residual_scale,
    make_inner_split,
    make_kfold_outer_splits,
    make_unique_run_dir,
    residualize_against_gene_mean,
    restore_from_gene_mean_residual,
    run_nested_selection,
    selectivity_labels_from_train,
    set_global_seed,
)
from embedding_provenance import (
    EMBEDDING_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    NAMES_RELATIVE_PATH,
    NORMALIZED_RELATIVE_PATH,
    validate_embedding_bundle,
)
from models import (
    DeepDepMLP,
    DepPLM,
    GNNRegressor,
    OneHotMLP,
    PLMMLP,
    normalized_adjacency,
)


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
DEPMAP = DATA / "depmap"
OUTPUT_ROOT = ROOT / "results" / "corrected_v2"
RESIDUAL_ANCHOR_OUTPUT_ROOT = ROOT / "results" / "residual_anchor_v1"


CANONICAL_CONDITIONS: dict[str, dict[str, Any]] = {
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

# A paired research matrix keeps architecture and preprocessing fixed while
# isolating the effect of the training-only gene-mean residual objective.
RESIDUAL_ANCHOR_CONDITIONS: dict[str, dict[str, Any]] = {
    "RA-PLM-absolute": {"model": "plmmlp", "target_mode": "absolute"},
    "RA-PLM-anchored": {
        "model": "plmmlp",
        "target_mode": "train_gene_mean_residual",
    },
    "RA-DepPLM-absolute": {"model": "depplm", "target_mode": "absolute"},
    "RA-DepPLM-anchored": {
        "model": "depplm",
        "target_mode": "train_gene_mean_residual",
    },
}

CONDITIONS: dict[str, dict[str, Any]] = {
    **CANONICAL_CONDITIONS,
    **RESIDUAL_ANCHOR_CONDITIONS,
}

FULL_GRID = [
    {"hidden": hidden, "lr": lr, "wd": 1e-5, "lamb_sel": 0.1}
    for hidden in (128, 256)
    for lr in (1e-3, 3e-4)
]

CANONICAL_SEEDS = (42, 43, 44)
TRANSFER_HOLDOUT_LINEAGES = ("Bone", "Kidney", "Myeloid")
TRANSFER_HOLDOUT_COUNTS = {"Bone": 35, "Kidney": 31, "Myeloid": 39}
CANONICAL_ARGUMENTS: dict[str, int | float] = {
    "folds": 5,
    "inner_val_fraction": 0.2,
    "inner_max_epochs": 20,
    "patience": 5,
    "batch_size": 64,
    "gene_subsample": 6_000,
    "top_mut": 5_000,
    "n_expr_pc": 128,
}


@dataclass
class BaseData:
    gene_effect: np.ndarray
    expression_raw: np.ndarray
    mutations_by_cell: list[set[str]]
    embeddings: np.ndarray
    edge_index: np.ndarray
    genes: list[str]
    cell_ids: list[str]
    lineages: np.ndarray
    source_cell_indices: np.ndarray
    source_gene_indices: np.ndarray
    expression_gene_names: list[str]


@dataclass
class FoldData:
    expression: np.ndarray
    mutation: MutationFeatures
    lineage: np.ndarray
    expression_fit: FittedExpressionPCA
    lineage_fit: FittedLineageEncoder
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    adjacency: torch.Tensor
    selectivity_labels: np.ndarray
    fit_indices: np.ndarray


@dataclass
class FinalArtifact:
    model: torch.nn.Module
    fold_data: FoldData
    config: dict[str, Any]
    gene_mean_reference: np.ndarray
    training: dict[str, Any]


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


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


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(int(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path = ROOT) -> dict[str, Any]:
    stat = path.stat()
    try:
        display_path = path.relative_to(relative_to)
    except ValueError:
        display_path = path
    return {
        "path": str(display_path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _atomic_write_outer_test_predictions(
    path: str | Path,
    *,
    prediction: np.ndarray,
    train_gene_mean: np.ndarray,
    outer_test_indices: np.ndarray,
    base: BaseData,
) -> dict[str, Any]:
    """Write the exact outer-test predictions and their row/column identities."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "prediction": np.ascontiguousarray(prediction, dtype=np.float32),
        "train_gene_mean": np.ascontiguousarray(train_gene_mean, dtype=np.float32),
        "outer_test_indices": np.ascontiguousarray(
            outer_test_indices, dtype=np.int64
        ),
        "source_cell_indices": np.ascontiguousarray(
            base.source_cell_indices[outer_test_indices], dtype=np.int64
        ),
        "source_gene_indices": np.ascontiguousarray(
            base.source_gene_indices, dtype=np.int64
        ),
        "cell_ids": np.asarray(
            [base.cell_ids[int(index)] for index in outer_test_indices], dtype=str
        ),
        "gene_ids": np.asarray(base.genes, dtype=str),
    }
    expected_shape = (outer_test_indices.size, len(base.genes))
    if arrays["prediction"].shape != expected_shape:
        raise ValueError(
            "prediction shape does not match the outer-test cell and gene order: "
            f"{arrays['prediction'].shape} != {expected_shape}"
        )
    if arrays["train_gene_mean"].shape != (len(base.genes),):
        raise ValueError("train_gene_mean must contain one value per output gene")
    if not np.isfinite(arrays["prediction"]).all():
        raise ValueError("refusing to persist non-finite outer-test predictions")
    if not np.isfinite(arrays["train_gene_mean"]).all():
        raise ValueError("refusing to persist a non-finite training gene mean")

    temporary = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
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

    file_record = _file_record(target, relative_to=target.parent)
    return {
        "schema": "DepPLM-outer-test-predictions-v1",
        "path": file_record["path"],
        "bytes": file_record["bytes"],
        "sha256": file_record["sha256"],
        "arrays": {
            name: {"dtype": str(array.dtype), "shape": list(array.shape)}
            for name, array in arrays.items()
        },
    }


def _source_stability_reasons(
    initial_records: Sequence[Mapping[str, Any]],
    paths: Sequence[Path],
) -> list[str]:
    """Return fail-closed reasons when an input or code file changed mid-run."""
    if len(initial_records) != len(paths):
        return ["source-record count changed during the run"]
    reasons: list[str] = []
    for initial, path in zip(initial_records, paths):
        current = _file_record(path)
        for field in ("path", "bytes", "sha256"):
            if current[field] != initial.get(field):
                reasons.append(
                    f"source changed during run: {initial.get('path', path)} ({field})"
                )
                break
    return reasons


def _load_expression(
    cell_ids: Sequence[str],
    allowed_genes: Sequence[str],
    max_columns: int | None,
) -> tuple[np.ndarray, list[str]]:
    path = DEPMAP / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    id_column = header[0]
    allowed = set(allowed_genes)
    selected_columns: list[str] = []
    selected_names: list[str] = []
    seen: set[str] = set()
    for column in header[1:]:
        symbol = column.split(" (")[0]
        if symbol in allowed and symbol not in seen:
            selected_columns.append(column)
            selected_names.append(symbol)
            seen.add(symbol)
            if max_columns is not None and len(selected_columns) >= max_columns:
                break
    if not selected_columns:
        raise RuntimeError("no expression columns map to the processed gene set")
    dtype = {column: np.float32 for column in selected_columns}
    frame = pd.read_csv(
        path,
        usecols=[id_column, *selected_columns],
        index_col=0,
        dtype=dtype,
    )
    missing = [cell_id for cell_id in cell_ids if cell_id not in frame.index]
    if missing:
        raise RuntimeError(f"expression is missing {len(missing)} requested cell lines")
    frame = frame.reindex(cell_ids)
    return frame.to_numpy(dtype=np.float32, copy=True), selected_names


def _load_mutations(cell_ids: Sequence[str]) -> list[set[str]]:
    wanted = set(cell_ids)
    by_cell: defaultdict[str, set[str]] = defaultdict(set)
    path = DEPMAP / "OmicsSomaticMutations.csv"
    for chunk in pd.read_csv(
        path,
        usecols=["HugoSymbol", "ModelID"],
        dtype=str,
        chunksize=250_000,
    ):
        chunk = chunk[chunk["ModelID"].isin(wanted)].dropna()
        for model_id, symbol in zip(chunk["ModelID"], chunk["HugoSymbol"]):
            if symbol:
                by_cell[str(model_id)].add(str(symbol))
    return [by_cell.get(cell_id, set()) for cell_id in cell_ids]


def _subgraph(edge_index: np.ndarray, selected_genes: np.ndarray, n_genes: int) -> np.ndarray:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.shape[1] != 2:
        raise ValueError("stored graph edge_index must have shape (E, 2)")
    remap = np.full(n_genes, -1, dtype=np.int64)
    remap[selected_genes] = np.arange(selected_genes.size, dtype=np.int64)
    keep = (remap[edges[:, 0]] >= 0) & (remap[edges[:, 1]] >= 0)
    mapped = np.stack([remap[edges[keep, 0]], remap[edges[keep, 1]]], axis=1)
    mapped.sort(axis=1)
    mapped = mapped[mapped[:, 0] != mapped[:, 1]]
    return np.unique(mapped, axis=0)


def load_base_data(*, pilot: bool, seed: int, pilot_cells: int, pilot_genes: int) -> BaseData:
    all_genes = _read_lines(PROCESSED / "genes.txt")
    embedding_names = _read_lines(ROOT / NAMES_RELATIVE_PATH)
    all_cells = _read_lines(PROCESSED / "cell_lines.txt")
    all_lineages = np.asarray(_read_lines(PROCESSED / "lineages.txt"), dtype=str)
    y_source = np.load(PROCESSED / "gene_effect.npy", mmap_mode="r")
    embedding_source = np.load(
        ROOT / EMBEDDING_RELATIVE_PATH, mmap_mode="r", allow_pickle=False
    )
    graph = np.load(PROCESSED / "ppi_graph.npz")
    full_edges = graph["edge_index"].astype(np.int64, copy=False)
    if y_source.shape != (len(all_genes), len(all_cells)):
        raise RuntimeError("processed labels do not align with genes/cell_lines manifests")
    if embedding_source.ndim != 2 or embedding_source.shape[0] != len(all_genes):
        raise RuntimeError("ESM embeddings do not align with the processed gene manifest")
    if embedding_names != all_genes:
        raise RuntimeError("ESM embedding names/order do not match genes.txt")

    if pilot:
        rng = np.random.default_rng(derive_seed(seed, "pilot-subset"))
        cell_idx = np.sort(
            rng.choice(len(all_cells), min(int(pilot_cells), len(all_cells)), replace=False)
        ).astype(np.int64)
        gene_idx = np.sort(
            rng.choice(len(all_genes), min(int(pilot_genes), len(all_genes)), replace=False)
        ).astype(np.int64)
        expression_limit = max(64, min(512, 2 * len(gene_idx)))
    else:
        cell_idx = np.arange(len(all_cells), dtype=np.int64)
        gene_idx = np.arange(len(all_genes), dtype=np.int64)
        expression_limit = None

    cells = [all_cells[int(idx)] for idx in cell_idx]
    genes = [all_genes[int(idx)] for idx in gene_idx]
    lineages = all_lineages[cell_idx]
    gene_effect = np.asarray(y_source[np.ix_(gene_idx, cell_idx)], dtype=np.float32)
    embeddings = np.asarray(embedding_source[gene_idx], dtype=np.float32)
    edges = _subgraph(full_edges, gene_idx, len(all_genes))

    print(
        f"[corrected] loading raw expression for {len(cells)} cells "
        f"({expression_limit or 'all mapped'} columns)",
        flush=True,
    )
    expression, expression_names = _load_expression(cells, all_genes, expression_limit)
    print(f"[corrected] loading raw mutations for {len(cells)} cells", flush=True)
    mutations = _load_mutations(cells)
    return BaseData(
        gene_effect=gene_effect,
        expression_raw=expression,
        mutations_by_cell=mutations,
        embeddings=embeddings,
        edge_index=edges,
        genes=genes,
        cell_ids=cells,
        lineages=lineages,
        source_cell_indices=cell_idx,
        source_gene_indices=gene_idx,
        expression_gene_names=expression_names,
    )


def fit_fold_data(
    base: BaseData,
    fit_indices: np.ndarray,
    *,
    seed: int,
    n_expr_pc: int,
    top_mut: int,
) -> FoldData:
    expression_fit = fit_expression_pca(
        base.expression_raw,
        fit_indices,
        n_components=n_expr_pc,
        seed=derive_seed(seed, "pca"),
    )
    expression = expression_fit.transform(base.expression_raw)
    mutation = fit_mutation_features(
        base.mutations_by_cell,
        base.genes,
        fit_indices,
        top_n=top_mut,
        k_tokens=K_TOKENS,
    )
    lineage_fit = fit_lineage_one_hot(base.lineages, fit_indices)
    lineage = lineage_fit.transform(base.lineages)
    edge_index_np = base.edge_index.T.astype(np.int64, copy=False)
    edge_weight_np = fit_comutation_edge_weights(
        edge_index_np,
        base.genes,
        mutation.matrix,
        mutation.vocabulary,
        fit_indices,
    )
    edge_index = torch.from_numpy(edge_index_np.copy()).long().contiguous()
    edge_weight = torch.from_numpy(edge_weight_np.copy()).float()
    adjacency = normalized_adjacency(
        edge_index, edge_weight, len(base.genes)
    )
    selectivity, _ = selectivity_labels_from_train(base.gene_effect, fit_indices)
    return FoldData(
        expression=expression,
        mutation=mutation,
        lineage=lineage,
        expression_fit=expression_fit,
        lineage_fit=lineage_fit,
        edge_index=edge_index,
        edge_weight=edge_weight,
        adjacency=adjacency,
        selectivity_labels=selectivity,
        fit_indices=np.asarray(fit_indices, dtype=np.int64).copy(),
    )


def preprocessing_manifest(
    fold_data: FoldData,
    evaluation_idx: np.ndarray,
) -> dict[str, Any]:
    unknown = np.isclose(fold_data.lineage[evaluation_idx].sum(axis=1), 0.0)
    return {
        "fit_indices_sha256": _hash_array(fold_data.fit_indices.astype("<i8")),
        "n_fit_cells": int(fold_data.fit_indices.size),
        "expression": {
            "input_features": int(fold_data.expression_fit.mean_.size),
            "output_components": int(fold_data.expression.shape[1]),
            "fitted_components": int(fold_data.expression_fit.components_.shape[0]),
            "mean_sha256": _hash_array(fold_data.expression_fit.mean_),
            "scale_sha256": _hash_array(fold_data.expression_fit.scale_),
            "components_sha256": _hash_array(fold_data.expression_fit.components_),
        },
        "mutation": {
            "vocabulary": list(fold_data.mutation.vocabulary),
            "vocabulary_sha256": _hash_strings(fold_data.mutation.vocabulary),
            "token_order_sha256": _hash_strings(fold_data.mutation.token_order),
            "n_vocabulary": len(fold_data.mutation.vocabulary),
            "n_token_genes": len(fold_data.mutation.token_order),
            "k_tokens": int(fold_data.mutation.tokens.shape[1]),
        },
        "graph": {
            "n_edges": int(fold_data.edge_index.shape[1]),
            "edge_weight_sha256": _hash_array(fold_data.edge_weight.numpy()),
            "edge_weight_mean": float(fold_data.edge_weight.mean().item())
            if fold_data.edge_weight.numel()
            else None,
            "n_non_neutral_weights": int(
                (~np.isclose(fold_data.edge_weight.numpy(), 1.0)).sum()
            ),
        },
        "lineage": {
            "categories": list(fold_data.lineage_fit.categories),
            "n_unknown_evaluation_cells": int(unknown.sum()),
            "unknown_encoding": "all_zero",
        },
        "selectivity_labels_sha256": _hash_array(fold_data.selectivity_labels),
    }


def build_model(
    config: Mapping[str, Any], base: BaseData, fold_data: FoldData
) -> torch.nn.Module:
    """Build a condition model while keeping embedding freezing comparable.

    A1-ESM changes only the source of the input embedding values: its random
    matrix is frozen exactly like the pretrained matrix used by full DepPLM.
    """
    hidden = int(config["hidden"])
    n_mut = fold_data.mutation.matrix.shape[1]
    n_expr = fold_data.expression.shape[1]
    n_lineage = fold_data.lineage.shape[1]
    n_genes = len(base.genes)
    model_type = str(config["model"])
    embeddings = torch.from_numpy(base.embeddings)
    if model_type == "deepdep":
        return DeepDepMLP(n_genes, n_mut, n_expr, n_lineage, hidden)
    if model_type == "onehot":
        return OneHotMLP(n_genes, n_mut, n_expr, n_lineage, hidden)
    if model_type == "plmmlp":
        return PLMMLP(embeddings, n_mut, n_expr, n_lineage, hidden)
    if model_type == "gnn":
        weighted = fold_data.edge_weight if config.get("comut", False) else None
        return GNNRegressor(
            embeddings,
            fold_data.adjacency,
            fold_data.edge_index,
            weighted,
            n_mut,
            n_expr,
            n_lineage,
            "gcn",
            hidden,
        )

    model_embeddings = embeddings
    if not config.get("use_pretrained", True):
        model_embeddings = torch.empty_like(embeddings)
        torch.nn.init.normal_(model_embeddings, mean=0.0, std=0.02)
    return DepPLM(
        model_embeddings,
        fold_data.adjacency,
        fold_data.edge_index,
        fold_data.edge_weight,
        n_mut,
        n_expr,
        n_lineage,
        hidden,
        k_tokens=K_TOKENS,
        use_pretrained=config.get("use_pretrained", True),
        use_gnn=config.get("use_gnn", True),
        use_comut=config.get("use_comut", True),
        use_multitask=config.get("use_multitask", True),
        use_crossattn=config.get("use_crossattn", True),
        freeze=True,
    )


def _forward(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    mutation: torch.Tensor,
    expression: torch.Tensor,
    lineage: torch.Tensor,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    gene_subset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    model_type = str(config["model"])
    if model_type in {"deepdep", "onehot", "plmmlp", "gnn"}:
        return model(mutation, expression, lineage, gene_subset=gene_subset), None
    prediction, selectivity, _ = model(
        mutation,
        expression,
        lineage,
        tokens[:, :K_TOKENS],
        token_mask[:, :K_TOKENS],
        gene_subset=gene_subset,
    )
    return prediction, selectivity


def _train_epochs(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    base: BaseData,
    fold_data: FoldData,
    train_idx: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    gene_subsample: int,
    device: str,
    seed: int,
    validation_idx: np.ndarray | None,
    patience: int,
    scheduler_horizon: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    epochs = int(epochs)
    horizon = epochs if scheduler_horizon is None else int(scheduler_horizon)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if horizon < epochs:
        raise ValueError("scheduler_horizon must be at least the number of trained epochs")
    set_global_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["lr"]),
        weight_decay=float(config["wd"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=horizon)
    rng = np.random.default_rng(derive_seed(seed, "batches"))
    train_gene_mean = fit_gene_mean_baseline(base.gene_effect, train_idx)
    target_mode = str(config.get("target_mode", "absolute"))
    if target_mode not in {"absolute", "train_gene_mean_residual"}:
        raise ValueError(f"unsupported target_mode: {target_mode!r}")
    best_score = -np.inf
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    use_amp = device.startswith("cuda")

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(train_idx)
        loss_sum = 0.0
        seen = 0
        for start in range(0, permutation.size, int(batch_size)):
            cell_idx = permutation[start : start + int(batch_size)]
            mutation = torch.from_numpy(fold_data.mutation.matrix[cell_idx]).to(device)
            expression = torch.from_numpy(fold_data.expression[cell_idx]).to(device)
            lineage = torch.from_numpy(fold_data.lineage[cell_idx]).to(device)
            tokens = torch.from_numpy(fold_data.mutation.tokens[cell_idx, :K_TOKENS]).to(device)
            token_mask = torch.from_numpy(
                fold_data.mutation.token_mask[cell_idx, :K_TOKENS]
            ).to(device)

            gene_subset = None
            if 0 < gene_subsample < base.gene_effect.shape[0]:
                selected = np.sort(
                    rng.choice(
                        base.gene_effect.shape[0], gene_subsample, replace=False
                    )
                ).astype(np.int64)
                gene_subset = torch.from_numpy(selected).to(device)
                target_np = base.gene_effect[np.ix_(selected, cell_idx)].T
                target_gene_mean = train_gene_mean[selected]
            else:
                target_np = base.gene_effect[:, cell_idx].T
                target_gene_mean = train_gene_mean
            if target_mode == "train_gene_mean_residual":
                target_np = residualize_against_gene_mean(
                    target_np,
                    target_gene_mean,
                )
            target = torch.from_numpy(np.asarray(target_np, dtype=np.float32)).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                prediction, selectivity = _forward(
                    model,
                    config,
                    mutation,
                    expression,
                    lineage,
                    tokens,
                    token_mask,
                    gene_subset,
                )
                finite = torch.isfinite(target)
                residual = torch.where(finite, prediction - target, torch.zeros_like(prediction))
                loss = residual.square().sum() / finite.sum().clamp(min=1)
                if selectivity is not None:
                    labels = torch.from_numpy(fold_data.selectivity_labels).to(device)
                    loss = loss + float(config.get("lamb_sel", 0.0)) * torch.nn.functional.binary_cross_entropy_with_logits(
                        selectivity, labels
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
        if validation_idx is not None:
            validation = evaluate_model(
                model,
                config,
                base,
                fold_data,
                validation_idx,
                batch_size=batch_size,
                device=device,
                include_spearman=False,
                gene_mean_reference=train_gene_mean,
                reference_scope="inner_train_gene_mean",
                fit_residual_scale=(target_mode == "train_gene_mean_residual"),
            )
            epoch_record["validation"] = validation
            score = float(validation["residual_r2_vs_train_gene_mean"])
            if not np.isfinite(score):
                raise RuntimeError(
                    "inner validation residual R2 is not finite; the training-mean "
                    "baseline has no evaluable variance"
                )
            if score > best_score + 1e-8:
                best_score = score
                best_epoch = epoch + 1
                best_metrics = validation
                bad_epochs = 0
            else:
                bad_epochs += 1
        history.append(epoch_record)
        print(
            f"[corrected] epoch={epoch + 1}/{epochs} "
            f"train_loss={epoch_record['train_loss']:.5f}"
            + (
                " val_residual_r2="
                f"{epoch_record['validation']['residual_r2_vs_train_gene_mean']:.5f}"
                if validation_idx is not None
                else ""
            ),
            flush=True,
        )
        if validation_idx is not None and bad_epochs >= int(patience):
            break

    training_record: dict[str, Any] = {
        "trained_epochs": len(history),
        "scheduler": "CosineAnnealingLR",
        "scheduler_horizon_epochs": int(horizon),
        "history": history,
        **({"target_mode": target_mode} if "target_mode" in config else {}),
    }
    if validation_idx is None:
        return model, training_record
    if best_metrics is None or best_epoch < 1:
        raise RuntimeError("inner validation did not produce a finite checkpoint")
    selected: dict[str, Any] = {
        **training_record,
        "selection_metric": "residual_r2_vs_train_gene_mean",
        "selection_score": float(best_score),
        "best_epoch": int(best_epoch),
        "best_validation_metrics": best_metrics,
    }
    if target_mode == "train_gene_mean_residual":
        selected["selected_config_updates"] = {
            "residual_scale": float(best_metrics["residual_scale"]),
        }
    return model, selected


def _safe_corr(prediction: np.ndarray, target: np.ndarray) -> float:
    finite = np.isfinite(prediction) & np.isfinite(target)
    if finite.sum() < 10:
        return np.nan
    x = prediction[finite]
    y = target[finite]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_model(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    base: BaseData,
    fold_data: FoldData,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: str,
    include_spearman: bool,
    gene_mean_reference: np.ndarray,
    reference_scope: str,
    return_predictions: bool = False,
    fit_residual_scale: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], np.ndarray]:
    from scipy.stats import rankdata

    model.eval()
    raw_prediction_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, indices.size, int(batch_size)):
            cell_idx = indices[start : start + int(batch_size)]
            mutation = torch.from_numpy(fold_data.mutation.matrix[cell_idx]).to(device)
            expression = torch.from_numpy(fold_data.expression[cell_idx]).to(device)
            lineage = torch.from_numpy(fold_data.lineage[cell_idx]).to(device)
            tokens = torch.from_numpy(fold_data.mutation.tokens[cell_idx, :K_TOKENS]).to(device)
            token_mask = torch.from_numpy(
                fold_data.mutation.token_mask[cell_idx, :K_TOKENS]
            ).to(device)
            prediction, _ = _forward(
                model,
                config,
                mutation,
                expression,
                lineage,
                tokens,
                token_mask,
            )
            predicted = prediction.float().cpu().numpy()
            target = np.asarray(base.gene_effect[:, cell_idx].T, dtype=np.float32)
            raw_prediction_chunks.append(predicted)
            target_chunks.append(target)
    raw_predictions = np.concatenate(raw_prediction_chunks, axis=0)
    all_targets = np.concatenate(target_chunks, axis=0)
    target_mode = str(config.get("target_mode", "absolute"))
    residual_scale: float | None = None
    if target_mode == "train_gene_mean_residual":
        residual_scale = (
            fit_validation_residual_scale(
                raw_predictions,
                all_targets,
                gene_mean_reference,
            )
            if fit_residual_scale
            else float(config.get("residual_scale", 1.0))
        )
        if not np.isfinite(residual_scale) or not 0.0 <= residual_scale <= 1.0:
            raise ValueError("residual_scale must be finite and between zero and one")
        all_predictions = restore_from_gene_mean_residual(
            raw_predictions * residual_scale,
            gene_mean_reference,
        )
    elif target_mode == "absolute":
        if fit_residual_scale:
            raise ValueError("residual scaling is only valid for residual target mode")
        all_predictions = raw_predictions
    else:
        raise ValueError(f"unsupported target_mode: {target_mode!r}")
    result: dict[str, Any] = {
        **dependency_metrics(all_predictions, all_targets, gene_mean_reference),
        "n_cells": int(indices.size),
        "reference_scope": str(reference_scope),
        "gene_mean_reference_sha256": _hash_array(gene_mean_reference),
        "raw_cellwise_correlation_role": "secondary_descriptive_only",
        **(
            {"prediction_target_mode": target_mode}
            if "target_mode" in config
            else {}
        ),
        **(
            {
                "residual_scale": residual_scale,
                "residual_scale_scope": (
                    "fit_on_current_inner_validation"
                    if fit_residual_scale
                    else "frozen_from_inner_validation"
                ),
            }
            if target_mode == "train_gene_mean_residual"
            else {}
        ),
    }
    if include_spearman:
        spearman: list[float] = []
        for row in range(all_predictions.shape[0]):
            mask = np.isfinite(all_predictions[row]) & np.isfinite(all_targets[row])
            if mask.sum() < 10:
                spearman.append(np.nan)
            else:
                spearman.append(
                    _safe_corr(
                        rankdata(all_predictions[row, mask]),
                        rankdata(all_targets[row, mask]),
                    )
                )
        spearman_array = np.asarray(spearman, dtype=np.float64)
        finite_spearman = spearman_array[np.isfinite(spearman_array)]
        result["cellwise_spearman_mean"] = (
            float(np.mean(finite_spearman)) if finite_spearman.size else float("nan")
        )
        result["cellwise_spearman_sd"] = (
            float(np.std(finite_spearman)) if finite_spearman.size else float("nan")
        )
        result["n_finite_cellwise_spearman"] = int(finite_spearman.size)
    if return_predictions:
        return result, np.ascontiguousarray(all_predictions, dtype=np.float32)
    return result


def _device_from_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return value


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


def _outer_splits(
    args: argparse.Namespace, base: BaseData
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if args.protocol == "p1":
        n_splits = 2 if args.pilot else int(args.folds)
        return [
            (f"fold-{fold_id:02d}", train, test)
            for fold_id, (train, test) in enumerate(
                make_kfold_outer_splits(len(base.cell_ids), n_splits, args.seed)
            )
        ]
    if args.pilot:
        raise ValueError("transfer pilot is disabled; use --protocol p1 for smoke testing")
    stored = json.loads((PROCESSED / "splits.json").read_text(encoding="utf-8"))
    p2_train = np.asarray(stored["P2_train"], dtype=np.int64)
    p2_test = np.asarray(stored["P2_test"], dtype=np.int64)
    _validate_transfer_split_semantics(base, p2_train, p2_test)
    if args.protocol == "transfer-forward":
        return [("complement-to-bkm", p2_train, p2_test)]
    return [("bkm-to-complement", p2_test, p2_train)]


def _validate_transfer_split_semantics(
    base: BaseData,
    complement_indices: np.ndarray,
    bkm_indices: np.ndarray,
) -> None:
    """Freeze the repository-defined BKM holdout without overinterpreting it."""
    n_cells = len(base.cell_ids)
    complement = np.asarray(complement_indices, dtype=np.int64)
    bkm = np.asarray(bkm_indices, dtype=np.int64)
    for name, values in (("complement", complement), ("BKM", bkm)):
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name} transfer indices must be a non-empty vector")
        if np.unique(values).size != values.size:
            raise ValueError(f"{name} transfer indices contain duplicates")
        if int(values.min()) < 0 or int(values.max()) >= n_cells:
            raise IndexError(f"{name} transfer indices are outside the cohort")
    if set(complement).intersection(bkm):
        raise ValueError("transfer complement and BKM cohorts overlap")
    if set(complement).union(bkm) != set(range(n_cells)):
        raise ValueError("transfer complement and BKM cohorts do not cover the cohort")
    observed = dict(sorted(Counter(str(value) for value in base.lineages[bkm]).items()))
    if observed != TRANSFER_HOLDOUT_COUNTS:
        raise ValueError(
            "repository transfer holdout no longer matches the frozen BKM definition: "
            f"observed {observed!r}, expected {TRANSFER_HOLDOUT_COUNTS!r}"
        )
    complement_lineages = set(str(value) for value in base.lineages[complement])
    unexpected = complement_lineages.intersection(TRANSFER_HOLDOUT_LINEAGES)
    if unexpected:
        raise ValueError(
            "BKM lineages leaked into the complementary transfer cohort: "
            f"{sorted(unexpected)!r}"
        )


def _assess_run_completion(
    *,
    pilot: bool,
    requested_fold_id: int,
    expected_fold_numbers: Sequence[int],
    completed_fold_numbers: Sequence[int],
    required_metric_values: Sequence[Mapping[str, Any]],
    canonical_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Fail closed unless a non-pilot run completed every planned outer fold."""
    expected = [int(value) for value in expected_fold_numbers]
    completed = [int(value) for value in completed_fold_numbers]
    reasons: list[str] = []
    if not expected:
        reasons.append("the planned outer-fold list is empty")
    if pilot:
        reasons.append("pilot mode is a structural smoke test")
    if requested_fold_id >= 0:
        reasons.append("--fold-id requests a partial outer-fold run")
    if completed != expected:
        reasons.append(
            f"completed outer folds {completed!r} do not match planned folds {expected!r}"
        )
    required_metrics = (
        "mse",
        "mae",
        "gene_mean_baseline_mse",
        "gene_mean_baseline_mae",
        "residual_r2_vs_train_gene_mean",
        "residual_cellwise_pearson_mean",
    )
    invalid_metrics: list[str] = []
    if len(required_metric_values) != len(completed):
        invalid_metrics.append("metric-record count does not match completed folds")
    for fold_number, metrics in zip(completed, required_metric_values):
        for metric in required_metrics:
            value = metrics.get(metric)
            if value is None or not np.isfinite(value):
                invalid_metrics.append(f"fold {fold_number} has non-finite {metric}")
    if invalid_metrics:
        reasons.append("; ".join(invalid_metrics))
    reasons.extend(str(reason) for reason in canonical_reasons)
    protocol_complete = not reasons
    return {
        "run_status": "protocol_complete" if protocol_complete else "partial",
        "protocol_complete": protocol_complete,
        "eligible_for_scientific_aggregation": protocol_complete,
        "scientific_result": False,
        "scientific_result_reason": (
            "This condition-by-seed run is eligible for a separately validated "
            "multi-condition, multi-seed aggregation. It is not by itself a scientific "
            "result."
            if protocol_complete
            else "; ".join(reasons)
        ),
        "planned_fold_numbers": expected,
        "completed_fold_numbers": completed,
        "required_finite_metrics": list(required_metrics),
    }


def _canonical_run_reasons(args: argparse.Namespace) -> list[str]:
    """Return deviations from the prespecified corrected-v2 campaign."""
    reasons: list[str] = []
    if bool(args.pilot):
        reasons.append("pilot mode is outside the canonical campaign")
    if int(args.fold_id) != -1:
        reasons.append("canonical runs require --fold-id -1")
    if int(args.seed) not in CANONICAL_SEEDS:
        reasons.append(f"seed {args.seed} is not one of {list(CANONICAL_SEEDS)}")
    for name, expected in CANONICAL_ARGUMENTS.items():
        observed = getattr(args, name)
        equal = (
            bool(np.isclose(float(observed), float(expected), rtol=0.0, atol=1e-12))
            if isinstance(expected, float)
            else int(observed) == int(expected)
        )
        if not equal:
            reasons.append(
                f"non-canonical --{name.replace('_', '-')}={observed}; expected {expected}"
            )
    return reasons


def run_fold(
    *,
    fold_name: str,
    fold_number: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    base: BaseData,
    condition: str,
    base_config: Mapping[str, Any],
    grid: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    device: str,
    run_dir: Path,
) -> dict[str, Any]:
    fold_seed = derive_seed(args.seed, fold_name)
    inner_train, inner_val = make_inner_split(
        outer_train,
        base.lineages,
        val_fraction=args.inner_val_fraction,
        seed=derive_seed(fold_seed, "inner-split"),
    )
    split_manifest = build_split_manifest(
        outer_train_idx=outer_train,
        inner_train_idx=inner_train,
        inner_val_idx=inner_val,
        outer_test_idx=outer_test,
        cell_ids=base.cell_ids,
        seed=fold_seed,
    )
    split_manifest.update(
        {
            "fold": fold_name,
            "protocol": args.protocol,
            "source_cell_indices": {
                "outer_train": [int(base.source_cell_indices[idx]) for idx in outer_train],
                "inner_train": [int(base.source_cell_indices[idx]) for idx in inner_train],
                "inner_validation": [int(base.source_cell_indices[idx]) for idx in inner_val],
                "outer_test": [int(base.source_cell_indices[idx]) for idx in outer_test],
            },
            "lineage_counts": {
                "outer_train": dict(
                    sorted(Counter(str(value) for value in base.lineages[outer_train]).items())
                ),
                "outer_test": dict(
                    sorted(Counter(str(value) for value in base.lineages[outer_test]).items())
                ),
            },
        }
    )
    atomic_write_json(run_dir / f"{fold_name}-split.json", split_manifest)

    print(f"[corrected] {fold_name}: fit inner-train preprocessing", flush=True)
    inner_data = fit_fold_data(
        base,
        inner_train,
        seed=derive_seed(fold_seed, "inner-preprocess"),
        n_expr_pc=args.n_expr_pc,
        top_mut=args.top_mut,
    )

    configurations = [{**base_config, **candidate} for candidate in grid]

    def tune_fn(config, train_idx, val_idx, seed):
        set_global_seed(seed)
        model = build_model(config, base, inner_data)
        _, tuning = _train_epochs(
            model,
            config,
            base,
            inner_data,
            train_idx,
            epochs=args.inner_max_epochs,
            batch_size=args.batch_size,
            gene_subsample=args.gene_subsample,
            device=device,
            seed=seed,
            validation_idx=val_idx,
            patience=args.patience,
            scheduler_horizon=args.inner_max_epochs,
        )
        return tuning

    def refit_fn(config, train_idx, epochs, seed):
        print(
            f"[corrected] {fold_name}: refit preprocessing on all outer-train cells",
            flush=True,
        )
        final_data = fit_fold_data(
            base,
            train_idx,
            seed=derive_seed(seed, "outer-preprocess"),
            n_expr_pc=args.n_expr_pc,
            top_mut=args.top_mut,
        )
        gene_mean_reference = fit_gene_mean_baseline(base.gene_effect, train_idx)
        set_global_seed(seed)
        model = build_model(config, base, final_data)
        model, training = _train_epochs(
            model,
            config,
            base,
            final_data,
            train_idx,
            epochs=epochs,
            batch_size=args.batch_size,
            gene_subsample=args.gene_subsample,
            device=device,
            seed=seed,
            validation_idx=None,
            patience=args.patience,
            scheduler_horizon=args.inner_max_epochs,
        )
        return FinalArtifact(
            model=model,
            fold_data=final_data,
            config=dict(config),
            gene_mean_reference=gene_mean_reference,
            training=training,
        )

    test_calls = 0
    prediction_record: dict[str, Any] | None = None

    def test_fn(artifact: FinalArtifact, test_idx: np.ndarray):
        nonlocal prediction_record, test_calls
        test_calls += 1
        if test_calls != 1:
            raise RuntimeError("outer test evaluation was invoked more than once")
        print(f"[corrected] {fold_name}: final outer-test evaluation", flush=True)
        evaluation = evaluate_model(
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
        metrics, predictions = evaluation
        prediction_record = _atomic_write_outer_test_predictions(
            run_dir / f"{fold_name}-predictions.npz",
            prediction=predictions,
            train_gene_mean=artifact.gene_mean_reference,
            outer_test_indices=test_idx,
            base=base,
        )
        return metrics

    started = time.time()
    selected = run_nested_selection(
        configs=configurations,
        outer_train_idx=outer_train,
        inner_train_idx=inner_train,
        inner_val_idx=inner_val,
        outer_test_idx=outer_test,
        tune_fn=tune_fn,
        refit_fn=refit_fn,
        test_fn=test_fn,
        seed=fold_seed,
    )
    final_artifact = selected.final_model
    if not isinstance(final_artifact, FinalArtifact):
        raise TypeError("refit did not return a FinalArtifact")
    if prediction_record is None:
        raise RuntimeError("outer-test prediction artifact was not written")
    final_data = final_artifact.fold_data
    result = {
        "fold": fold_name,
        "fold_number": int(fold_number),
        "condition": condition,
        "protocol": args.protocol,
        "seed": int(args.seed),
        "fold_seed": int(fold_seed),
        "best_config": dict(selected.best_config),
        "best_epoch": int(selected.best_epoch),
        "selection": {
            "metric": "residual_r2_vs_train_gene_mean",
            "maximize": True,
            "score": float(selected.selection_score),
        },
        "optimization": {
            "scheduler": final_artifact.training["scheduler"],
            "scheduler_horizon_epochs": int(
                final_artifact.training["scheduler_horizon_epochs"]
            ),
            "refit_epochs": int(final_artifact.training["trained_epochs"]),
            "selection_epoch_budget": int(args.inner_max_epochs),
        },
        "tuning": selected.tuning_results,
        "test_metrics": dict(selected.test_metrics),
        "outer_test_predictions": prediction_record,
        "test_evaluation_calls": int(test_calls),
        "preprocessing": {
            "inner_train_fit": preprocessing_manifest(inner_data, inner_val),
            "outer_train_refit": preprocessing_manifest(final_data, outer_test),
        },
        "wall_seconds": time.time() - started,
        "pilot": bool(args.pilot),
        "scientific_result": False,
        "result_scope": "single_outer_fold",
    }
    atomic_write_json(run_dir / f"{fold_name}-result.json", result)
    del selected.final_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _int_at_least(minimum: int):
    """Build an argparse converter that rejects integers below ``minimum``."""

    def convert(value: str) -> int:
        parsed = int(value)
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return parsed

    return convert


def _inner_val_fraction(value: str) -> float:
    """Parse the inner-validation fraction required by ``make_inner_split``."""
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 < parsed < 0.5:
        raise argparse.ArgumentTypeError(
            "must be finite and strictly between 0 and 0.5"
        )
    return parsed


def _fold_id(value: str) -> int:
    parsed = int(value)
    if parsed < -1:
        raise argparse.ArgumentTypeError("must be -1 or a non-negative fold index")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=["p1", "transfer-forward", "transfer-reverse"],
        default="p1",
    )
    parser.add_argument("--condition", choices=sorted(CONDITIONS), default="DepPLM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=_int_at_least(2), default=5)
    parser.add_argument("--fold-id", type=_fold_id, default=-1)
    parser.add_argument(
        "--inner-val-fraction", type=_inner_val_fraction, default=0.2
    )
    parser.add_argument("--inner-max-epochs", type=_int_at_least(1), default=20)
    parser.add_argument("--patience", type=_int_at_least(1), default=5)
    parser.add_argument("--batch-size", type=_int_at_least(1), default=64)
    parser.add_argument("--gene-subsample", type=_int_at_least(0), default=6_000)
    parser.add_argument("--top-mut", type=_int_at_least(1), default=5_000)
    parser.add_argument("--n-expr-pc", type=_int_at_least(1), default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-cells", type=_int_at_least(1), default=96)
    parser.add_argument("--pilot-genes", type=_int_at_least(1), default=128)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.pilot:
        args.inner_max_epochs = 1
        args.patience = 1
        args.batch_size = min(args.batch_size, 32)
        args.gene_subsample = 0
        args.top_mut = min(args.top_mut, 128)
        args.n_expr_pc = min(args.n_expr_pc, 32)
    device = _device_from_arg(args.device)
    set_global_seed(args.seed)
    embedding_provenance = validate_embedding_bundle(ROOT)
    research_condition = args.condition in RESIDUAL_ANCHOR_CONDITIONS
    output_root = RESIDUAL_ANCHOR_OUTPUT_ROOT if research_condition else OUTPUT_ROOT
    run_dir = make_unique_run_dir(
        output_root,
        protocol=args.protocol + ("-pilot" if args.pilot else ""),
        condition=args.condition,
        seed=args.seed,
    )
    run_started = time.time()
    input_paths = [
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
    ]
    if args.protocol != "p1":
        input_paths.append(PROCESSED / "splits.json")
    code_paths = [
        Path(__file__).resolve(),
        ROOT / "code" / "corrected_protocol.py",
        ROOT / "code" / "models.py",
        ROOT / "code" / "03_embed_proteins.py",
        ROOT / "code" / "embedding_provenance.py",
    ]
    input_records = [_file_record(path) for path in input_paths]
    code_records = [_file_record(path) for path in code_paths]
    cuda_available = bool(torch.cuda.is_available())
    atomic_write_json(
        run_dir / "run-manifest.json",
        {
            "schema": "DepPLM-corrected-v2.2",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": sys.argv,
            "arguments": vars(args),
            "device": device,
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
            },
            "runtime": {
                "amp_enabled": device.startswith("cuda"),
                "amp_precision": "bfloat16" if device.startswith("cuda") else None,
                "cuda_available": cuda_available,
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "cuda_device_name": torch.cuda.get_device_name(0)
                if device == "cuda"
                else None,
                "cuda_device_capability": list(torch.cuda.get_device_capability(0))
                if device == "cuda"
                else None,
                "deterministic_algorithms_enabled": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "deterministic_debug_mode": int(
                    torch.get_deterministic_debug_mode()
                ),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            },
            "git": _git_record(),
            "inputs": input_records,
            "code": code_records,
            "embedding_provenance": embedding_provenance,
            **(
                {
                    "research_design": {
                        "campaign": "residual_anchor_v1",
                        "status": "unvalidated_hypothesis",
                        "paired_factors": [
                            "target_mode",
                            "validation_residual_scale",
                        ],
                        "target_mode": CONDITIONS[args.condition].get(
                            "target_mode", "absolute"
                        ),
                        "claim_policy": (
                            "Do not claim improvement until the complete paired matrix is "
                            "independently aggregated from outer-test predictions."
                        ),
                    }
                }
                if research_condition
                else {}
            ),
            "excluded_conditions": {
                "C-FEAT": (
                    "Excluded from corrected_v2 because the legacy handcrafted "
                    "feature artifact contains expression mean and standard deviation "
                    "computed over the complete cell-line cohort. Its so-called GC "
                    "feature is the fraction of G and C amino-acid letters in a protein "
                    "sequence, not nucleotide GC content."
                )
            },
            "output_policy": "unique_run_directory_atomic_json_no_overwrite",
            "pilot": bool(args.pilot),
            "run_status": "in_progress",
            "protocol_complete": False,
            "eligible_for_scientific_aggregation": False,
            "scientific_result": False,
            "scientific_result_reason": (
                "The run manifest is written before data loading and training. "
                "Only a later completion marker can establish protocol completion."
            ),
        },
    )

    print(f"[corrected] output={run_dir}", flush=True)
    print(f"[corrected] device={device} pilot={args.pilot}", flush=True)
    base = load_base_data(
        pilot=args.pilot,
        seed=args.seed,
        pilot_cells=args.pilot_cells,
        pilot_genes=args.pilot_genes,
    )
    atomic_write_json(
        run_dir / "cohort-manifest.json",
        {
            "n_genes": len(base.genes),
            "n_cells": len(base.cell_ids),
            "n_expression_features_raw": int(base.expression_raw.shape[1]),
            "n_string_edges": int(base.edge_index.shape[0]),
            "genes_sha256": _hash_strings(base.genes),
            "cell_ids_sha256": _hash_strings(base.cell_ids),
            "expression_gene_names_sha256": _hash_strings(base.expression_gene_names),
            "source_cell_indices": base.source_cell_indices,
            "source_gene_indices": base.source_gene_indices,
        },
    )

    base_config = dict(CONDITIONS[args.condition])
    if args.pilot:
        grid: list[Mapping[str, Any]] = [
            {"hidden": 32, "lr": 1e-3, "wd": 1e-5, "lamb_sel": 0.1}
        ]
    else:
        grid = FULL_GRID
    outer_splits = _outer_splits(args, base)
    if args.fold_id >= 0:
        if args.fold_id >= len(outer_splits):
            raise IndexError(
                f"--fold-id {args.fold_id} is outside 0..{len(outer_splits) - 1}"
            )
        selected_splits = [(args.fold_id, outer_splits[args.fold_id])]
    elif args.pilot:
        selected_splits = [(0, outer_splits[0])]
    else:
        selected_splits = list(enumerate(outer_splits))

    fold_results = []
    for fold_number, (fold_name, outer_train, outer_test) in selected_splits:
        fold_results.append(
            run_fold(
                fold_name=fold_name,
                fold_number=fold_number,
                outer_train=outer_train,
                outer_test=outer_test,
                base=base,
                condition=args.condition,
                base_config=base_config,
                grid=grid,
                args=args,
                device=device,
                run_dir=run_dir,
            )
        )

    metric_names = [
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
    ]
    summary_metrics: dict[str, dict[str, float | int | None]] = {}
    for metric in metric_names:
        values = np.asarray(
            [result["test_metrics"].get(metric, np.nan) for result in fold_results],
            dtype=np.float64,
        )
        finite_values = values[np.isfinite(values)]
        summary_metrics[metric] = {
            "mean": float(np.mean(finite_values)) if finite_values.size else None,
            "sd": float(np.std(finite_values)) if finite_values.size else None,
            "n": int(finite_values.size),
        }
    source_stability_reasons = [
        *_source_stability_reasons(input_records, input_paths),
        *_source_stability_reasons(code_records, code_paths),
    ]
    completion = _assess_run_completion(
        pilot=bool(args.pilot),
        requested_fold_id=int(args.fold_id),
        expected_fold_numbers=list(range(len(outer_splits))),
        completed_fold_numbers=[int(result["fold_number"]) for result in fold_results],
        required_metric_values=[result["test_metrics"] for result in fold_results],
        canonical_reasons=[*_canonical_run_reasons(args), *source_stability_reasons],
    )
    fold_artifacts = []
    for result in fold_results:
        fold_name = str(result["fold"])
        for suffix, extension in (
            ("result", "json"),
            ("split", "json"),
            ("predictions", "npz"),
        ):
            fold_artifacts.append(
                _file_record(
                    run_dir / f"{fold_name}-{suffix}.{extension}", relative_to=run_dir
                )
            )
    summary = {
        "schema": "DepPLM-corrected-v2.2",
        "protocol": args.protocol,
        "condition": args.condition,
        **(
            {
                "campaign": "residual_anchor_v1",
                "target_mode": base_config.get("target_mode", "absolute"),
            }
            if research_condition
            else {}
        ),
        "seed": int(args.seed),
        "pilot": bool(args.pilot),
        "n_completed_folds": len(fold_results),
        "fold_files": [f"{result['fold']}-result.json" for result in fold_results],
        "primary_metric": "residual_r2_vs_train_gene_mean",
        "selection_policy": (
            "maximize inner-validation residual R2 against inner-training gene means; "
            "report outer-test metrics against outer-training gene means"
        ),
        "metrics": summary_metrics,
        "wall_seconds": time.time() - run_started,
        "outer_test_policy": "exactly_once_per_fold_after_fixed_epoch_refit",
        "canonical_campaign": {
            "seeds": list(CANONICAL_SEEDS),
            "arguments": CANONICAL_ARGUMENTS,
            "tuning_grid": FULL_GRID,
        },
        "source_stability": {
            "checked_after_training": True,
            "unchanged": not source_stability_reasons,
            "reasons": source_stability_reasons,
        },
        "artifacts": fold_artifacts,
        **completion,
    }
    summary_path = run_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    if completion["protocol_complete"]:
        atomic_write_json(
            run_dir / "completion.json",
            {
                "schema": "DepPLM-corrected-v2.2-completion",
                "run_status": "protocol_complete",
                "summary_file": summary_path.name,
                "summary_sha256": _sha256_file(summary_path),
                "run_manifest_sha256": _sha256_file(run_dir / "run-manifest.json"),
                "cohort_manifest_sha256": _sha256_file(
                    run_dir / "cohort-manifest.json"
                ),
                "artifacts": fold_artifacts,
                "eligible_for_scientific_aggregation": True,
                "scientific_result": False,
                "required_next_step": (
                    "Validate and aggregate the prespecified conditions and seeds before "
                    "using this run for manuscript conclusions."
                ),
            },
        )
    print(f"[corrected] complete: {run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
