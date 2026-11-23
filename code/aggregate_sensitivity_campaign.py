#!/usr/bin/env python3
"""Aggregate a five-fold diagnostic campaign (single split/training seed).

Used for the identity-threshold sensitivity runs (20/30/40/50%) and for
exploratory embedding variants. Each gene is predicted exactly once across the
five outer folds, so the consensus is the single out-of-fold prediction.

Outputs a JSON report and a consensus-evidence.npz compatible with the primary
diagnostic tooling.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

METHOD_KEY = "family_distance_stack_gene_mean_prediction"
BASELINE_KEY = "job_selected_baseline_gene_mean_prediction"
GLOBAL_KEY = "global_train_mean_prediction"


def _metrics(prediction: np.ndarray, target: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(prediction) & np.isfinite(target) & np.isfinite(reference)
    if not np.any(finite):
        return {"n": 0, "status": "undefined_empty"}
    p, y, g = prediction[finite], target[finite], reference[finite]
    error = p - y
    denominator = float(np.sum((g - y) ** 2))
    pearson = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else None
    return {
        "n": int(finite.sum()),
        "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(np.abs(error))),
        "r2": None if denominator <= 0 else float(1.0 - np.sum(error ** 2) / denominator),
        "pearson": pearson,
    }


def _component_groups(component_ids: np.ndarray) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index, value in enumerate(component_ids.astype(str).tolist()):
        grouped.setdefault(value, []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]


def _bootstrap_delta(method, baseline, global_prediction, target, component_ids, replicates, confidence, seed):
    groups = _component_groups(component_ids)
    global_loss = np.square(global_prediction - target)
    method_loss = np.square(method - target)
    baseline_loss = np.square(baseline - target)
    sums_gain = np.asarray([np.sum((global_loss - method_loss)[g]) for g in groups])
    sums_delta = np.asarray([np.sum((baseline_loss - method_loss)[g]) for g in groups])
    sums_den = np.asarray([np.sum(global_loss[g]) for g in groups])
    sizes = np.asarray([g.size for g in groups], dtype=np.int64)
    rng = np.random.default_rng(seed)
    boot_gain = np.empty(replicates)
    boot_delta = np.empty(replicates)
    chunk = max(1, 500_000 // max(1, len(groups)))
    for start in range(0, replicates, chunk):
        stop = min(replicates, start + chunk)
        sampled = rng.integers(0, len(groups), size=(stop - start, len(groups)))
        count = np.sum(sizes[sampled], axis=1)
        den = np.sum(sums_den[sampled], axis=1)
        boot_gain[start:stop] = np.sum(sums_gain[sampled], axis=1) / count
        boot_delta[start:stop] = np.divide(
            np.sum(sums_delta[sampled], axis=1), den,
            out=np.full(stop - start, np.nan), where=den > 0,
        )
    alpha = (1.0 - confidence) / 2.0
    denominator = float(np.sum(global_loss))
    return {
        "n_components": len(groups),
        "replicates": replicates,
        "method_mse_improvement_vs_global": {
            "estimate": float(np.mean(global_loss - method_loss)),
            "ci_low": float(np.nanquantile(boot_gain, alpha)),
            "ci_high": float(np.nanquantile(boot_gain, 1 - alpha)),
        },
        "delta_r2_method_minus_baseline": {
            "estimate": None if denominator <= 0 else float(np.sum(baseline_loss - method_loss) / denominator),
            "ci_low": float(np.nanquantile(boot_delta, alpha)),
            "ci_high": float(np.nanquantile(boot_delta, 1 - alpha)),
        },
    }


def _stratum(mask, arrays, replicates, confidence, seed):
    index = np.flatnonzero(mask)
    if index.size < 3:
        return {"status": "undefined_too_few_genes", "n": int(index.size)}
    target = arrays["observed_gene_mean"][index].astype(np.float64)
    method = arrays[METHOD_KEY][index].astype(np.float64)
    baseline = arrays[BASELINE_KEY][index].astype(np.float64)
    global_prediction = arrays[GLOBAL_KEY][index].astype(np.float64)
    components = arrays["component_ids"][index]
    output = {
        "n_genes": int(index.size),
        "n_components": int(len(set(components.astype(str).tolist()))),
        "method": _metrics(method, target, global_prediction),
        "selected_baseline": _metrics(baseline, target, global_prediction),
        "bootstrap": _bootstrap_delta(
            method, baseline, global_prediction, target, components,
            replicates, confidence, seed,
        ),
    }
    if output["method"].get("r2") is not None and output["selected_baseline"].get("r2") is not None:
        output["delta_r2_point"] = output["method"]["r2"] - output["selected_baseline"]["r2"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()

    run_dirs = sorted(path.parent for path in args.runs_root.glob("*/completion.json"))
    if not run_dirs:
        raise SystemExit("no completed run directories found")
    args.outdir.mkdir(parents=True, exist_ok=True)

    records = []
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "run-manifest.json").read_text())
        arguments = manifest["arguments"]
        records.append({
            "run_id": manifest["run_id"],
            "fold": int(arguments["outer_fold"]),
            "split_seed": int(arguments["split_seed"]),
            "training_seed": int(arguments["training_seed"]),
            "identity": int(arguments["identity_percent"]),
            "path": str(run_dir),
        })
    folds = sorted(row["fold"] for row in records)
    if folds != [0, 1, 2, 3, 4]:
        raise SystemExit(f"expected folds 0-4, found {folds}")
    if len({(row["split_seed"], row["training_seed"]) for row in records}) != 1:
        raise SystemExit("runs must share one split seed and one training seed")

    # Determine full gene count from the canonical indices.
    max_index = -1
    for row in records:
        with np.load(Path(row["path"]) / "predictions.npz", allow_pickle=False) as handle:
            max_index = max(max_index, int(handle["outer_test_gene_indices"].max()))
    n_genes = max_index + 1

    keys = [
        "observed_gene_mean", METHOD_KEY, BASELINE_KEY, GLOBAL_KEY,
        "consensus_nearest_training_distance", "component_ids", "gene_ids",
        "component_size_full_release", "protein_length_residues",
    ]
    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        if key == "component_ids":
            arrays[key] = np.full(n_genes, "", dtype="<U32")
        elif key == "gene_ids":
            arrays[key] = np.full(n_genes, "", dtype="<U32")
        else:
            arrays[key] = np.full(n_genes, np.nan, dtype=np.float64)
    arrays["component_size_full_release"] = np.zeros(n_genes, dtype=np.int64)
    arrays["protein_length_residues"] = np.zeros(n_genes, dtype=np.int64)
    filled = np.zeros(n_genes, dtype=bool)

    for row in records:
        with np.load(Path(row["path"]) / "predictions.npz", allow_pickle=False) as handle:
            index = handle["outer_test_gene_indices"].astype(np.int64)
            if np.any(filled[index]):
                raise SystemExit(f"duplicate gene prediction across folds for {row['run_id']}")
            filled[index] = True
            arrays["gene_ids"][index] = handle["outer_test_gene_ids"]
            arrays["component_ids"][index] = handle["outer_test_component_ids"]
            arrays["component_size_full_release"][index] = handle["outer_test_component_size_full_release"]
            arrays["protein_length_residues"][index] = handle["outer_test_protein_length_residues"]
            arrays["observed_gene_mean"][index] = handle["observed_gene_mean"]
            arrays[METHOD_KEY][index] = handle[METHOD_KEY]
            arrays[BASELINE_KEY][index] = handle[BASELINE_KEY]
            arrays[GLOBAL_KEY][index] = handle[GLOBAL_KEY]
            arrays["consensus_nearest_training_distance"][index] = handle["nearest_outer_train_cosine_distance"]
    if not np.all(filled):
        raise SystemExit(f"missing predictions for {int((~filled).sum())} genes")

    target = arrays["observed_gene_mean"].astype(np.float64)
    method = arrays[METHOD_KEY].astype(np.float64)
    baseline = arrays[BASELINE_KEY].astype(np.float64)
    global_prediction = arrays[GLOBAL_KEY].astype(np.float64)
    components = arrays["component_ids"].astype(str)
    distance = arrays["consensus_nearest_training_distance"].astype(np.float64)

    report: dict[str, Any] = {
        "label": args.label,
        "status": "diagnostic_complete",
        "runs": records,
        "n_genes": int(n_genes),
        "headline": {
            "method": _metrics(method, target, global_prediction),
            "selected_baseline": _metrics(baseline, target, global_prediction),
        },
        "bootstrap": _bootstrap_delta(
            method, baseline, global_prediction, target, components,
            args.replicates, args.confidence, args.seed,
        ),
    }
    report["headline"]["delta_r2_point"] = (
        report["headline"]["method"]["r2"] - report["headline"]["selected_baseline"]["r2"]
    )
    edges = np.quantile(distance, np.linspace(0, 1, 5))
    labels = np.clip(np.digitize(distance, edges[1:-1], right=False), 0, 3)
    report["distance_quartiles"] = {
        "edges": [float(x) for x in edges],
        "quartiles": {
            f"Q{i + 1}": _stratum(
                labels == i, arrays, args.replicates, args.confidence, args.seed + i,
            )
            for i in range(4)
        },
    }

    with (args.outdir / "consensus-evidence.npz").open("wb") as handle:
        np.savez_compressed(handle, **{key: value for key, value in arrays.items()})
    (args.outdir / "report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, default=lambda x: float(x) if isinstance(x, np.floating) else x),
        encoding="utf-8",
    )
    print(json.dumps({
        "label": args.label,
        "method_r2": report["headline"]["method"]["r2"],
        "baseline_r2": report["headline"]["selected_baseline"]["r2"],
        "delta_r2": report["headline"]["delta_r2_point"],
        "delta_ci": report["bootstrap"]["delta_r2_method_minus_baseline"],
        "n_genes": n_genes,
    }, indent=1))


if __name__ == "__main__":
    main()
