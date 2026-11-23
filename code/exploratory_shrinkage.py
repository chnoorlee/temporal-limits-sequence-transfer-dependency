#!/usr/bin/env python3
"""Exploratory shrinkage, wider-grid and relaxed-constraint probes.

Five outer folds from the diagnostic campaign (split seed 42, training seed 42).
All hyperparameters and shrinkage factors are fitted on outer-training genes
only; outer-test labels are used only for scoring. These are post hoc probes,
not replacements for the registered campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors


def _component_fold(component: str, folds: int = 3) -> int:
    digest = hashlib.sha256(component.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _r2(prediction, target, reference):
    denominator = float(np.sum((reference - target) ** 2))
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.sum((prediction - target) ** 2) / denominator)


def _knn_weights(distances: np.ndarray, temperature: float) -> np.ndarray:
    logits = -distances / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def _ridge_fit(x, y, alpha):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    xtx = z.T @ z
    xtx.flat[:: xtx.shape[0] + 1] += alpha
    beta = np.linalg.solve(xtx, z.T @ (y - y.mean()))
    return mean, scale, beta, float(y.mean())


def _ridge_predict(model, x):
    mean, scale, beta, intercept = model
    return ((x - mean) / scale) @ beta + intercept


def _relaxed_stack(train_parts, y, z_train, test_parts, z_test, lambdas):
    best = None
    for lam in lambdas:
        def objective(theta):
            a = theta[:3]
            b = theta[3:]
            stacked = np.zeros_like(y)
            for index, key in enumerate(("G", "K", "R")):
                stacked = stacked + (a[index] + b[index] * z_train) * train_parts[key][:, 0]
            residual = stacked - y
            return float(np.mean(residual ** 2) + lam * np.sum(b ** 2))

        constraints = [
            {"type": "eq", "fun": lambda theta: np.sum(theta[:3]) - 1.0},
            {"type": "eq", "fun": lambda theta: np.sum(theta[3:])},
        ]
        result = minimize(
            objective, np.asarray([1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0]),
            method="SLSQP", constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            continue
        if best is None or result.fun < best[0]:
            best = (float(result.fun), result.x, lam)
    if best is None:
        raise RuntimeError("relaxed stack failed")
    _, theta, lam = best
    a, b = theta[:3], theta[3:]
    prediction = np.zeros(test_parts["G"].shape[0])
    for index, key in enumerate(("G", "K", "R")):
        prediction += (a[index] + b[index] * z_test) * test_parts[key][:, 0]
    return prediction, {"a": a.tolist(), "b": b.tolist(), "lambda": lam, "n_iterations": int(result.nit)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--gene-effect", type=Path, required=True)
    parser.add_argument("--consensus-tsv", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    genes = [line.strip() for line in args.genes.read_text().splitlines() if line.strip()]
    embedding = np.asarray(np.load(args.embedding, mmap_mode="r"), dtype=np.float64)
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    effect = np.asarray(np.load(args.gene_effect, mmap_mode="r"), dtype=np.float64).T
    with np.errstate(invalid="ignore"):
        target = np.nanmean(effect, axis=0)
    components = {}
    with args.consensus_tsv.open() as handle:
        header = next(handle).rstrip("\n").split("\t")
        gene_index = header.index("gene")
        component_index = header.index("component_id")
        for line in handle:
            values = line.rstrip("\n").split("\t")
            components[values[gene_index]] = values[component_index]
    component_array = np.asarray([components.get(gene, f"single-{gene}") for gene in genes])

    run_dirs = sorted(args.runs_root.glob("*-ss42-ts42"))
    if len(run_dirs) != 5:
        raise SystemExit(f"expected five folds, found {len(run_dirs)}")
    n = len(genes)
    predictions = {key: np.full(n, np.nan) for key in (
        "official_method", "official_baseline", "official_global",
        "wide_knn", "js_knn", "wide_ridge", "relaxed_stack",
    )}
    folds_report = []
    for run_dir in run_dirs:
        with np.load(run_dir / "predictions.npz", allow_pickle=False) as handle:
            test = handle["outer_test_gene_indices"].astype(np.int64)
            official = {
                "official_method": handle["family_distance_stack_gene_mean_prediction"],
                "official_baseline": handle["job_selected_baseline_gene_mean_prediction"],
                "official_global": handle["global_train_mean_prediction"],
            }
        train = np.ones(n, dtype=bool)
        train[test] = False
        train &= np.isfinite(target)
        test = test[np.isfinite(target[test])]
        y_train = target[train]
        global_train = float(y_train.mean())
        fold_ids = np.asarray([_component_fold(component_array[index]) for index in np.flatnonzero(train)])
        # KNN with leave-one-out on outer-training genes.
        neighbors = NearestNeighbors(n_neighbors=201, metric="cosine").fit(embedding[train])
        dist_train, idx_train = neighbors.kneighbors(embedding[train])
        dist_train = dist_train[:, 1:]
        idx_train = idx_train[:, 1:]
        best = None
        for k in (5, 20, 50, 100, 200):
            for temperature in (0.01, 0.03, 0.1, 0.3, 1.0):
                weights = _knn_weights(dist_train[:, :k], temperature)
                prediction = np.sum(weights * y_train[idx_train[:, :k]], axis=1)
                mse = float(np.mean((prediction - y_train) ** 2))
                if best is None or mse < best[0]:
                    best = (mse, k, temperature, prediction)
        _, k, temperature, loo_knn = best
        dist_test, idx_test = neighbors.kneighbors(embedding[test])
        weights_test = _knn_weights(dist_test[:, :k], temperature)
        knn_test = np.sum(weights_test * y_train[idx_test[:, :k]], axis=1)
        # James-Stein-style scalar shrinkage of KNN toward the global mean.
        centered_knn = loo_knn - global_train
        centered_y = y_train - global_train
        denominator = float(np.dot(centered_knn, centered_knn))
        c = float(np.dot(centered_knn, centered_y) / denominator) if denominator > 0 else 0.0
        c = float(np.clip(c, 0.0, 1.5))
        js_test = global_train + c * (knn_test - global_train)
        # Ridge with three-fold component CV on outer training.
        ridge_best = None
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
            errors = []
            for fold in range(3):
                fit = fold_ids != fold
                model = _ridge_fit(embedding[train][fit], y_train[fit], alpha)
                pred = _ridge_predict(model, embedding[train][~fit])
                errors.append(np.mean((pred - y_train[~fit]) ** 2))
            score = float(np.mean(errors))
            if ridge_best is None or score < ridge_best[0]:
                ridge_best = (score, alpha)
        alpha = ridge_best[1]
        ridge_model = _ridge_fit(embedding[train], y_train, alpha)
        ridge_test = _ridge_predict(ridge_model, embedding[test])
        # Relaxed stack on out-of-fold training predictions.
        oof_ridge = np.empty_like(y_train)
        for fold in range(3):
            fit = fold_ids != fold
            model = _ridge_fit(embedding[train][fit], y_train[fit], alpha)
            oof_ridge[~fit] = _ridge_predict(model, embedding[train][~fit])
        train_parts = {
            "G": np.full((y_train.size, 1), global_train),
            "K": loo_knn[:, None],
            "R": oof_ridge[:, None],
        }
        test_parts = {
            "G": np.full((test.size, 1), global_train),
            "K": knn_test[:, None],
            "R": ridge_test[:, None],
        }
        d_train = dist_train[:, 0]
        d_test = dist_test[:, 0]
        q10, q90 = np.quantile(d_train, [0.1, 0.9])
        z_train = np.clip((d_train - q10) / (q90 - q10), 0.0, 1.0)
        z_test = np.clip((d_test - q10) / (q90 - q10), 0.0, 1.0)
        stack_test, stack_info = _relaxed_stack(
            train_parts, y_train, z_train, test_parts, z_test, (0.0, 0.01, 0.1)
        )
        for key, values in official.items():
            predictions[key][test] = values
        predictions["wide_knn"][test] = knn_test
        predictions["js_knn"][test] = js_test
        predictions["wide_ridge"][test] = ridge_test
        predictions["relaxed_stack"][test] = stack_test
        folds_report.append({
            "run_id": run_dir.name, "k": k, "temperature": temperature,
            "js_c": c, "ridge_alpha": alpha, "stack": stack_info,
        })

    report = {"folds": folds_report, "methods": {}}
    reference = predictions["official_global"]
    for key, values in predictions.items():
        finite = np.isfinite(values) & np.isfinite(target) & np.isfinite(reference)
        report["methods"][key] = {
            "n": int(finite.sum()),
            "r2": _r2(values[finite], target[finite], reference[finite]),
            "mse": float(np.mean((values[finite] - target[finite]) ** 2)),
        }
    (args.outdir / "exploratory_shrinkage.json").write_text(
        json.dumps(report, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report["methods"], indent=1))


if __name__ == "__main__":
    main()
