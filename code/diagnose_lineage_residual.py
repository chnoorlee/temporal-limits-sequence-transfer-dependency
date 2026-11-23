#!/usr/bin/env python3
"""Lineage-stratified residual probe.

For each cancer lineage with enough models, centre the lineage-specific gene mean
on the pan-context gene mean and ask whether the frozen ESM-2 embedding predicts
those deviations under the same component-disjoint outer folds. A near-zero or
negative R² means sequence does not capture within-lineage context beyond the
gene mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _r2_zero(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.sum(target ** 2))
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.sum((prediction - target) ** 2) / denominator)


def _ridge(train_x, train_y, test_x, alpha):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    x = (train_x - mean) / scale
    xtx = x.T @ x
    xtx.flat[:: xtx.shape[0] + 1] += alpha
    beta = np.linalg.solve(xtx, x.T @ (train_y - train_y.mean()))
    return (test_x - mean) / scale @ beta + train_y.mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-effect", type=Path, required=True)
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--lineages", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--minimum-cells", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=100.0)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    genes = [line.strip() for line in args.genes.read_text().splitlines() if line.strip()]
    lineages = np.asarray([line.strip() for line in args.lineages.read_text().splitlines() if line.strip()])
    effect = np.load(args.gene_effect, mmap_mode="r")
    effect = np.asarray(effect, dtype=np.float32, order="C").T  # cells x genes
    embedding = np.asarray(np.load(args.embedding, mmap_mode="r"), dtype=np.float64)
    if embedding.shape[0] != len(genes):
        raise SystemExit("embedding rows do not match genes")

    with np.errstate(invalid="ignore"):
        gene_mean = np.nanmean(effect.astype(np.float64), axis=0)
    folds: list[np.ndarray] = []
    for run_dir in sorted(args.runs_root.glob("*-ss42-ts42/predictions.npz")):
        with np.load(run_dir, allow_pickle=False) as handle:
            folds.append(handle["outer_test_gene_indices"].astype(np.int64))
    if len(folds) != 5:
        raise SystemExit(f"expected five folds, found {len(folds)}")

    report = {"alpha": args.alpha, "minimum_cells": args.minimum_cells, "lineages": {}}
    for lineage in sorted(set(lineages.tolist())):
        mask = lineages == lineage
        cells = int(mask.sum())
        if cells < args.minimum_cells:
            continue
        with np.errstate(invalid="ignore"):
            lineage_mean = np.nanmean(effect[mask].astype(np.float64), axis=0)
        centered = lineage_mean - gene_mean
        finite = np.isfinite(centered) & np.isfinite(gene_mean)
        predictions = np.full(len(genes), np.nan)
        for test in folds:
            test = test[finite[test]]
            train = np.ones(len(genes), dtype=bool)
            train[test] = False
            train &= finite
            predictions[test] = _ridge(
                embedding[train], centered[train], embedding[test], args.alpha
            )
        valid = np.isfinite(predictions) & finite
        report["lineages"][lineage] = {
            "n_cells": cells,
            "n_genes": int(valid.sum()),
            "r2_zero": _r2_zero(predictions[valid], centered[valid]),
            "centered_sd": float(np.std(centered[valid])),
        }

    values = [entry["r2_zero"] for entry in report["lineages"].values()]
    report["summary"] = {
        "n_lineages": len(values),
        "mean_r2": float(np.nanmean(values)) if values else None,
        "median_r2": float(np.nanmedian(values)) if values else None,
        "max_r2": float(np.nanmax(values)) if values else None,
        "fraction_r2_above_0.01": float(np.mean(np.asarray(values) > 0.01)) if values else None,
    }
    (args.outdir / "lineage_residual_probe.json").write_text(
        json.dumps(report, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
