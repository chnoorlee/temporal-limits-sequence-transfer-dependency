"""Reproduce a training-only per-gene mean baseline for the legacy splits.

This audit is deliberately model-free.  For every evaluation split it estimates
one score per gene from training cells only, then assigns those scores to every
test cell.  The resulting metrics quantify how much of the manuscript's
per-cell, across-gene performance can be obtained without cell context.

The command writes one new JSON artifact below
``results/corrected_v2/audits``.  Existing files are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = ROOT / "data" / "processed" / "gene_effect.npy"
DEFAULT_SPLITS = ROOT / "data" / "processed" / "splits.json"
DEFAULT_LINEAGES = ROOT / "data" / "processed" / "lineages.txt"
OUTPUT_DIR = ROOT / "results" / "corrected_v2" / "audits"
TRANSFER_HOLDOUT_COUNTS = {"Bone": 35, "Kidney": 31, "Myeloid": 39}


class OutputExistsError(FileExistsError):
    """Raised when an audit would overwrite an existing artifact."""


class SourceChangedError(RuntimeError):
    """Raised when an audited input or the analysis source changes mid-run."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bound_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read once so the recorded hash describes the exact bytes subsequently parsed."""
    content = path.read_bytes()
    return content, {
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _verify_source_records(records: Iterable[dict[str, Any]]) -> None:
    for record in records:
        path = Path(str(record["path"]))
        if not path.is_file():
            raise SourceChangedError(f"audited source disappeared during analysis: {path}")
        observed_size = path.stat().st_size
        observed_hash = file_sha256(path)
        if observed_size != record["bytes"] or observed_hash != record["sha256"]:
            raise SourceChangedError(f"audited source changed during analysis: {path}")


def verify_audit_sources(payload: dict[str, Any]) -> None:
    """Recheck all recorded sources immediately before publishing an audit."""
    records = [payload["analysis_code"], *payload["inputs"].values()]
    _verify_source_records(records)


def _as_indices(values: Iterable[int], n_cells: int, name: str) -> np.ndarray:
    indices = np.asarray(list(values), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional index list")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} contains duplicate cell indices")
    if indices.min() < 0 or indices.max() >= n_cells:
        raise ValueError(f"{name} contains an out-of-range cell index")
    return indices


def training_gene_means(labels: np.ndarray, train_indices: Iterable[int]) -> np.ndarray:
    """Estimate each gene's mean using finite labels in training cells only."""
    if labels.ndim != 2:
        raise ValueError("labels must have shape (genes, cells)")
    train = _as_indices(train_indices, labels.shape[1], "train_indices")
    train_labels = labels[:, train]
    finite = np.isfinite(train_labels)
    counts = finite.sum(axis=1)
    sums = np.where(finite, train_labels, 0.0).sum(axis=1, dtype=np.float64)
    means = np.full(labels.shape[0], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including exact tie handling."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denominator == 0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def evaluate_gene_scores(
    labels: np.ndarray,
    test_indices: Iterable[int],
    gene_scores: np.ndarray,
) -> dict[str, Any]:
    """Apply fixed gene scores to test cells using the repository metric axes."""
    if labels.ndim != 2:
        raise ValueError("labels must have shape (genes, cells)")
    scores = np.asarray(gene_scores, dtype=np.float64)
    if scores.shape != (labels.shape[0],):
        raise ValueError("gene_scores must contain exactly one score per gene")
    test = _as_indices(test_indices, labels.shape[1], "test_indices")

    pearson_values: list[float] = []
    spearman_values: list[float] = []
    absolute_error_sum = 0.0
    observed_pairs = 0
    evaluable_cells = 0

    for cell_index in test:
        truth = labels[:, cell_index]
        mask = np.isfinite(truth) & np.isfinite(scores)
        if int(mask.sum()) < 10:
            continue
        predicted = scores[mask]
        observed = truth[mask].astype(np.float64, copy=False)
        pearson_values.append(_pearson(predicted, observed))
        spearman_values.append(
            _pearson(_average_ranks(predicted), _average_ranks(observed))
        )
        absolute_error_sum += float(np.abs(predicted - observed).sum())
        observed_pairs += int(mask.sum())
        evaluable_cells += 1

    if evaluable_cells == 0 or observed_pairs == 0:
        raise ValueError("no test cell has at least 10 finite gene labels")

    pearson_array = np.asarray(pearson_values, dtype=np.float64)
    spearman_array = np.asarray(spearman_values, dtype=np.float64)
    return {
        "n_test_cells": int(test.size),
        "n_evaluable_cells": evaluable_cells,
        "n_observed_pairs": observed_pairs,
        "pearson_mean_across_cells": float(np.nanmean(pearson_array)),
        "pearson_sd_across_cells": float(np.nanstd(pearson_array)),
        "spearman_mean_across_cells": float(np.nanmean(spearman_array)),
        "mae_across_observed_pairs": absolute_error_sum / observed_pairs,
    }


def make_random_folds(n_cells: int, n_splits: int, seed: int) -> list[np.ndarray]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_cells < n_splits:
        raise ValueError("n_cells must be at least n_splits")
    permutation = np.random.default_rng(seed).permutation(n_cells)
    return [part.astype(np.int64, copy=False) for part in np.array_split(permutation, n_splits)]


def audit_p1(labels: np.ndarray, n_splits: int = 5, seed: int = 42) -> dict[str, Any]:
    folds = make_random_folds(labels.shape[1], n_splits, seed)
    fold_results: list[dict[str, Any]] = []

    for fold_id, test_indices in enumerate(folds):
        train_indices = np.concatenate(
            [folds[i] for i in range(n_splits) if i != fold_id]
        )
        scores = training_gene_means(labels, train_indices)
        metrics = evaluate_gene_scores(labels, test_indices, scores)
        fold_results.append(
            {
                "fold": fold_id,
                "n_train_cells": int(train_indices.size),
                **metrics,
            }
        )

    aggregate_keys = (
        "pearson_mean_across_cells",
        "spearman_mean_across_cells",
        "mae_across_observed_pairs",
    )
    aggregate = {
        key: float(np.mean([fold[key] for fold in fold_results]))
        for key in aggregate_keys
    }
    aggregate["pearson_sd_across_fold_means"] = float(
        np.std([fold["pearson_mean_across_cells"] for fold in fold_results])
    )
    return {
        "protocol": "P1_random_cell_folds",
        "seed": seed,
        "n_splits": n_splits,
        "folds": fold_results,
        "aggregate_mean_of_fold_metrics": aggregate,
    }


def _lineage_counts(lineages: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in lineages[indices]).items()))


def audit_p2(
    labels: np.ndarray,
    split_manifest: dict[str, Any],
    lineages: np.ndarray,
) -> dict[str, Any]:
    if lineages.shape != (labels.shape[1],):
        raise ValueError("lineages must contain exactly one label per cell")
    source = _as_indices(split_manifest["P2_train"], labels.shape[1], "P2_train")
    target = _as_indices(split_manifest["P2_test"], labels.shape[1], "P2_test")
    if np.intersect1d(source, target).size:
        raise ValueError("P2_train and P2_test must be disjoint")
    observed = np.sort(np.concatenate((source, target)))
    expected = np.arange(labels.shape[1], dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError("P2_train and P2_test must cover every cell exactly once")

    target_counts = _lineage_counts(lineages, target)
    if target_counts != TRANSFER_HOLDOUT_COUNTS:
        raise ValueError(
            "P2_test must contain exactly the frozen BKM holdout composition "
            f"{TRANSFER_HOLDOUT_COUNTS}; observed {target_counts}"
        )
    source_counts = _lineage_counts(lineages, source)
    unexpected_holdouts = {
        lineage: source_counts[lineage]
        for lineage in TRANSFER_HOLDOUT_COUNTS
        if source_counts.get(lineage, 0) != 0
    }
    if unexpected_holdouts:
        raise ValueError(
            "P2_train must be the BKM-free complement; observed holdout lineages "
            f"{unexpected_holdouts}"
        )

    directions: dict[str, Any] = {}
    for name, train_indices, test_indices in (
        ("P2_train_to_P2_test", source, target),
        ("P2_test_to_P2_train", target, source),
    ):
        scores = training_gene_means(labels, train_indices)
        directions[name] = {
            "n_train_cells": int(train_indices.size),
            "train_lineage_counts": _lineage_counts(lineages, train_indices),
            "test_lineage_counts": _lineage_counts(lineages, test_indices),
            **evaluate_gene_scores(labels, test_indices, scores),
        }
    return {
        "protocol": "P2_repository_defined_mixed_lineage_split",
        "interpretation": (
            "P2_test contains Myeloid, Bone, and Kidney cells, while P2_train is a "
            "mixed-lineage complement. These directions are not pure solid-to-"
            "haematologic or haematologic-to-solid transfers."
        ),
        "directions": directions,
    }


def build_audit(
    labels_path: Path,
    splits_path: Path,
    lineages_path: Path = DEFAULT_LINEAGES,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    labels_path = labels_path.resolve()
    splits_path = splits_path.resolve()
    lineages_path = lineages_path.resolve()
    analysis_path = Path(__file__).resolve()

    analysis_bytes, analysis_record = _read_bound_bytes(analysis_path)
    labels_bytes, labels_record = _read_bound_bytes(labels_path)
    splits_bytes, splits_record = _read_bound_bytes(splits_path)
    lineages_bytes, lineages_record = _read_bound_bytes(lineages_path)

    labels = np.load(io.BytesIO(labels_bytes), allow_pickle=False)
    if labels.ndim != 2:
        raise ValueError("gene-effect input must have shape (genes, cells)")
    split_manifest = json.loads(splits_bytes.decode("utf-8"))
    lineages = np.asarray(
        [line.strip() for line in lineages_bytes.decode("utf-8").splitlines()],
        dtype=str,
    )
    p1 = audit_p1(labels, n_splits=n_splits, seed=seed)
    p2 = audit_p2(labels, split_manifest, lineages)

    source_records = [
        analysis_record,
        labels_record,
        splits_record,
        lineages_record,
    ]
    _verify_source_records(source_records)

    payload = {
        "audit": "training-only per-gene mean baseline",
        "interpretation": (
            "Each test cell receives the gene means estimated from its training cells. "
            "The baseline uses no mutation, expression, lineage, graph, or sequence input."
        ),
        "metric_axis": "within each test cell across genes, matching code/run_experiments.py:evaluate",
        "analysis_code": analysis_record,
        "inputs": {
            "gene_effect": {
                **labels_record,
                "shape_genes_by_cells": [int(labels.shape[0]), int(labels.shape[1])],
            },
            "splits": splits_record,
            "lineages": lineages_record,
        },
        "source_stability": {
            "input_hash_binding": "sha256_of_exact_bytes_parsed",
            "paths_unchanged_after_analysis": True,
            "paths_rechecked_immediately_before_publication": False,
        },
        "p1": p1,
        "p2": p2,
    }
    del analysis_bytes
    return payload


def write_json_no_overwrite(payload: dict[str, Any], target: Path) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise OutputExistsError(f"refusing to overwrite existing output: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--lineages", type=Path, default=DEFAULT_LINEAGES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--output-name",
        help="JSON filename below results/corrected_v2/audits (default derives from seed/folds)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_name = args.output_name or f"gene_mean_baseline_seed{args.seed}_{args.folds}fold.json"
    if Path(output_name).name != output_name or not output_name.endswith(".json"):
        raise ValueError("output-name must be a plain .json filename")
    target = OUTPUT_DIR / output_name
    if target.exists():
        raise OutputExistsError(f"refusing to overwrite existing output: {target}")

    payload = build_audit(
        args.labels,
        args.splits,
        args.lineages,
        n_splits=args.folds,
        seed=args.seed,
    )
    verify_audit_sources(payload)
    payload["source_stability"][
        "paths_rechecked_immediately_before_publication"
    ] = True
    write_json_no_overwrite(payload, target)

    p1 = payload["p1"]["aggregate_mean_of_fold_metrics"]
    directions = payload["p2"]["directions"]
    print(f"wrote {target}")
    print(
        "P1 gene-mean baseline: "
        f"Pearson={p1['pearson_mean_across_cells']:.8f}, "
        f"Spearman={p1['spearman_mean_across_cells']:.8f}, "
        f"MAE={p1['mae_across_observed_pairs']:.8f}"
    )
    for name, metrics in directions.items():
        print(
            f"{name}: Pearson={metrics['pearson_mean_across_cells']:.8f}, "
            f"Spearman={metrics['spearman_mean_across_cells']:.8f}, "
            f"MAE={metrics['mae_across_observed_pairs']:.8f}"
        )


if __name__ == "__main__":
    main()
