#!/usr/bin/env python3
"""Exploratory probe of a minimal orthogonal sequence feature.

Asks whether log protein length adds predictive signal beyond the frozen ESM-2
embedding under the same component-disjoint outer splits. Four predictors are
compared per job on outer-test genes:

  G        outer-training global mean
  L        OLS on log protein length
  E        ridge on standardized ESM-2 embeddings (alpha = 1)
  E+L      ridge on embeddings plus log length (alpha = 1)

This is a post hoc descriptive probe, not a replacement for the nested baseline
selection of the primary campaign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _read_fasta_lengths(path: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    name = None
    count = 0
    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    lengths[name] = count
                name = line[1:].split("|", 1)[0].strip()
                count = 0
            else:
                count += len(line)
    if name is not None:
        lengths[name] = count
    return lengths


def _r2(prediction: np.ndarray, target: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.sum((reference - target) ** 2))
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.sum((prediction - target) ** 2) / denominator)


def _ridge_fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> np.ndarray:
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
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--gene-effect", type=Path, required=True)
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    genes = [line.strip() for line in args.genes.read_text().splitlines() if line.strip()]
    lengths = _read_fasta_lengths(args.fasta)
    log_length = np.asarray([np.log(lengths[gene]) for gene in genes], dtype=np.float64)
    effect = np.load(args.gene_effect, mmap_mode="r")
    with np.errstate(invalid="ignore"):
        target = np.nanmean(np.asarray(effect, dtype=np.float64), axis=1)
    embedding = np.load(args.embedding, mmap_mode="r")
    if embedding.shape[0] != len(genes):
        raise SystemExit("embedding row count does not match genes")
    embedding = np.asarray(embedding, dtype=np.float64)
    finite_target = np.isfinite(target)

    run_dirs = sorted(path.parent for path in args.runs_root.glob("*/predictions.npz"))
    jobs = []
    for run_dir in run_dirs:
        with np.load(run_dir / "predictions.npz", allow_pickle=False) as handle:
            test = handle["outer_test_gene_indices"].astype(np.int64)
        train_mask = np.ones(len(genes), dtype=bool)
        train_mask[test] = False
        train_mask &= finite_target
        test = test[finite_target[test]]
        y_train = target[train_mask]
        y_test = target[test]
        global_prediction = np.full(test.size, y_train.mean())
        length_prediction = _ridge_fit_predict(
            log_length[train_mask, None], y_train, log_length[test, None], 0.0
        )
        embedding_prediction = _ridge_fit_predict(
            embedding[train_mask], y_train, embedding[test], args.alpha
        )
        combined_prediction = _ridge_fit_predict(
            np.column_stack([embedding[train_mask], log_length[train_mask]]),
            y_train,
            np.column_stack([embedding[test], log_length[test]]),
            args.alpha,
        )
        jobs.append({
            "run_id": run_dir.name,
            "n_test": int(test.size),
            "global_r2": _r2(global_prediction, y_test, global_prediction),
            "length_r2": _r2(length_prediction, y_test, global_prediction),
            "embedding_r2": _r2(embedding_prediction, y_test, global_prediction),
            "embedding_plus_length_r2": _r2(combined_prediction, y_test, global_prediction),
        })

    rho = float(spearmanr(log_length[finite_target], target[finite_target]).statistic)
    summary = {
        "status": "diagnostic_complete",
        "n_jobs": len(jobs),
        "alpha": args.alpha,
        "spearman_log_length_vs_gene_mean": rho,
        "mean_job_r2": {
            key: float(np.nanmean([job[key] for job in jobs]))
            for key in ("length_r2", "embedding_r2", "embedding_plus_length_r2")
        },
        "median_job_r2": {
            key: float(np.nanmedian([job[key] for job in jobs]))
            for key in ("length_r2", "embedding_r2", "embedding_plus_length_r2")
        },
        "mean_delta_length_added": float(np.nanmean([
            job["embedding_plus_length_r2"] - job["embedding_r2"] for job in jobs
        ])),
        "jobs": jobs,
    }
    (args.outdir / "orthogonal_feature_probe.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "n_jobs": summary["n_jobs"],
        "spearman_log_length_vs_gene_mean": rho,
        "mean_length_r2": summary["mean_job_r2"]["length_r2"],
        "mean_embedding_r2": summary["mean_job_r2"]["embedding_r2"],
        "mean_embedding_plus_length_r2": summary["mean_job_r2"]["embedding_plus_length_r2"],
        "mean_delta": summary["mean_delta_length_added"],
    }, indent=1))


if __name__ == "__main__":
    main()
