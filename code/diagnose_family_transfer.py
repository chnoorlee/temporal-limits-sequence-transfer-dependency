#!/usr/bin/env python3
"""Post-review diagnostics for the frozen family-transfer design.

Consumes an aggregated campaign directory (consensus-evidence.npz and
campaign-metrics.json) plus the per-job run manifests, and produces:

1. distance-quartile / decile performance for FAMILY-DISTANCE-STACK, the
   per-job selected baseline and the consensus global reference;
2. component-size and protein-length strata with paired bootstrap intervals;
3. per-job learned distance-stack coefficients (a_m, b_m), solver status and
   constraint-binding diagnostics.

Diagnostic outputs only. They never replace the frozen primary campaign.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import spearmanr


METHOD_KEY = "family_distance_stack_gene_mean_prediction"
BASELINE_KEY = "consensus_job_selected_baseline_gene_mean_prediction"
GLOBAL_KEY = "global_train_mean_prediction"
METHOD_ORDER = ("GLOBAL-TRAIN-MEAN", "SEQUENCE-WEIGHTED-KNN", "PLM-RIDGE")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _metrics(prediction: np.ndarray, target: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    finite = np.isfinite(prediction) & np.isfinite(target) & np.isfinite(reference)
    if not np.any(finite):
        return {"n": 0, "status": "undefined_empty"}
    p = prediction[finite]
    y = target[finite]
    g = reference[finite]
    error = p - y
    denominator = float(np.sum((g - y) ** 2))
    pearson = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else None
    spearman_value = float(spearmanr(p, y).statistic) if np.std(p) > 0 and np.std(y) > 0 else None
    slope = None
    intercept = None
    if np.std(p) > 0:
        design = np.column_stack([np.ones_like(p), p])
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        intercept, slope = float(intercept), float(slope)
    return {
        "n": int(finite.sum()),
        "mse": float(np.mean(error ** 2)),
        "mae": float(np.mean(np.abs(error))),
        "r2": None if denominator <= 0 else float(1.0 - np.sum(error ** 2) / denominator),
        "pearson": pearson,
        "spearman": spearman_value,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def _component_groups(component_ids: np.ndarray) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index, value in enumerate(component_ids.astype(str).tolist()):
        grouped.setdefault(value, []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]


def _bootstrap_delta(
    method: np.ndarray,
    baseline: np.ndarray,
    global_prediction: np.ndarray,
    target: np.ndarray,
    component_ids: np.ndarray,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    groups = _component_groups(component_ids)
    global_loss = np.square(global_prediction - target)
    method_loss = np.square(method - target)
    baseline_loss = np.square(baseline - target)
    sums_method_gain = np.asarray([np.sum((global_loss - method_loss)[g]) for g in groups])
    sums_delta = np.asarray([np.sum((baseline_loss - method_loss)[g]) for g in groups])
    sums_den = np.asarray([np.sum(global_loss[g]) for g in groups])
    sizes = np.asarray([g.size for g in groups], dtype=np.int64)
    rng = np.random.default_rng(seed)
    boot_method_gain = np.empty(replicates)
    boot_delta = np.empty(replicates)
    chunk = max(1, 500_000 // max(1, len(groups)))
    for start in range(0, replicates, chunk):
        stop = min(replicates, start + chunk)
        sampled = rng.integers(0, len(groups), size=(stop - start, len(groups)))
        count = np.sum(sizes[sampled], axis=1)
        den = np.sum(sums_den[sampled], axis=1)
        boot_method_gain[start:stop] = np.sum(sums_method_gain[sampled], axis=1) / count
        boot_delta[start:stop] = np.divide(
            np.sum(sums_delta[sampled], axis=1), den,
            out=np.full(stop - start, np.nan), where=den > 0,
        )
    alpha = (1.0 - confidence) / 2.0
    gain_low, gain_high = np.nanquantile(boot_method_gain, [alpha, 1 - alpha])
    delta_low, delta_high = np.nanquantile(boot_delta, [alpha, 1 - alpha])
    denominator = float(np.sum(global_loss))
    return {
        "n_components": len(groups),
        "replicates": replicates,
        "method_mse_improvement_vs_global": {
            "estimate": float(np.mean(global_loss - method_loss)),
            "ci_low": float(gain_low),
            "ci_high": float(gain_high),
        },
        "delta_r2_method_minus_baseline": {
            "estimate": None if denominator <= 0 else float(np.sum(baseline_loss - method_loss) / denominator),
            "ci_low": float(delta_low),
            "ci_high": float(delta_high),
        },
    }


def _stratum_report(
    mask: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    index = np.flatnonzero(mask)
    if index.size < 3:
        return {"status": "undefined_too_few_genes", "n": int(index.size)}
    target = arrays["observed_gene_mean"][index].astype(np.float64)
    method = arrays[METHOD_KEY][index].astype(np.float64)
    baseline = arrays[BASELINE_KEY][index].astype(np.float64)
    global_prediction = arrays[GLOBAL_KEY][index].astype(np.float64)
    components = arrays["component_ids"][index]
    report = {
        "n_genes": int(index.size),
        "n_components": int(len(set(components.astype(str).tolist()))),
        "global_reference": _metrics(global_prediction, target, global_prediction),
        "method": _metrics(method, target, global_prediction),
        "selected_baseline": _metrics(baseline, target, global_prediction),
        "method_minus_baseline": _metrics(method, target, baseline),
        "bootstrap": _bootstrap_delta(
            method, baseline, global_prediction, target, components,
            replicates=replicates, confidence=confidence, seed=seed,
        ),
    }
    if report["method"].get("r2") is not None and report["selected_baseline"].get("r2") is not None:
        report["delta_r2_point"] = report["method"]["r2"] - report["selected_baseline"]["r2"]
    return report


def _quantile_strata(values: np.ndarray, bins: int) -> tuple[np.ndarray, list[float]]:
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    labels = np.clip(np.digitize(values, edges[1:-1], right=False), 0, bins - 1)
    return labels, [float(x) for x in edges]


def _walk_fits(node: Any, path: str = "") -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(node, dict):
        if {"a", "b", "method_order", "solver_status"} <= set(node):
            yield path, node
        for key, value in node.items():
            yield from _walk_fits(value, f"{path}/{key}")
    elif isinstance(node, list):
        for position, value in enumerate(node):
            yield from _walk_fits(value, f"{path}[{position}]")


def _binding_flags(fit: Mapping[str, Any]) -> dict[str, Any]:
    a = [float(x) for x in fit["a"]]
    b = [float(x) for x in fit["b"]]
    residuals = fit.get("constraint_residuals", {})
    tol = 1e-8
    return {
        "a_global": a[0], "a_knn": a[1], "a_ridge": a[2],
        "b_global": b[0], "b_knn": b[1], "b_ridge": b[2],
        "endpoint_weight_at_zero": min(a) <= tol,
        "global_slope_at_zero": abs(b[0]) <= tol,
        "knn_slope_at_zero": abs(b[1]) <= tol,
        "ridge_slope_at_zero": abs(b[2]) <= tol,
        "sum_a_minus_one": float(residuals.get("abs_sum_a_minus_one", float("nan"))),
        "sum_b": float(residuals.get("abs_sum_b", float("nan"))),
        "lambda": float(fit.get("gate_lambda", float("nan"))),
        "solver_success": bool(fit.get("solver_success", False)),
        "solver_status": int(fit.get("solver_status", -999)),
        "solver_iterations": int(fit.get("solver_iterations", -1)),
        "solver_message": str(fit.get("solver_message", "")),
        "objective_value": float(fit.get("objective_value", float("nan"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--locked-metrics", type=Path)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    with np.load(args.campaign_dir / "consensus-evidence.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    campaign_metrics = json.loads((args.campaign_dir / "campaign-metrics.json").read_text())
    target = arrays["observed_gene_mean"].astype(np.float64)
    n = target.size

    report: dict[str, Any] = {
        "status": "diagnostic_complete",
        "n_genes": int(n),
        "seed": args.seed,
        "replicates": args.replicates,
        "headline": {
            "method": _metrics(arrays[METHOD_KEY], target, arrays[GLOBAL_KEY]),
            "selected_baseline": _metrics(arrays[BASELINE_KEY], target, arrays[GLOBAL_KEY]),
            "delta_r2_point": None,
        },
    }
    report["headline"]["delta_r2_point"] = (
        report["headline"]["method"]["r2"] - report["headline"]["selected_baseline"]["r2"]
    )
    if args.locked_metrics and args.locked_metrics.is_file():
        locked = json.loads(args.locked_metrics.read_text())["development"]
        report["locked_reference"] = {
            "method_r2": locked["proposed"]["r2_vs_consensus_global_train_mean"],
            "baseline_r2": locked["selected_baseline"]["r2_vs_consensus_global_train_mean"],
            "delta_r2": locked["delta_r2"]["estimate"],
            "method_r2_abs_diff": abs(
                report["headline"]["method"]["r2"] - locked["proposed"]["r2_vs_consensus_global_train_mean"]
            ),
            "baseline_r2_abs_diff": abs(
                report["headline"]["selected_baseline"]["r2"] - locked["selected_baseline"]["r2_vs_consensus_global_train_mean"]
            ),
            "delta_r2_abs_diff": abs(
                report["headline"]["delta_r2_point"] - locked["delta_r2"]["estimate"]
            ),
        }

    distance = arrays["consensus_nearest_training_cosine_distance"].astype(np.float64)
    quartile_labels, quartile_edges = _quantile_strata(distance, 4)
    decile_labels, decile_edges = _quantile_strata(distance, 10)
    report["distance_strata"] = {
        "quartile_edges": quartile_edges,
        "decile_edges": decile_edges,
        "quartiles": {
            f"Q{i + 1}": _stratum_report(
                quartile_labels == i, arrays,
                replicates=args.replicates, confidence=args.confidence, seed=args.seed + i,
            )
            for i in range(4)
        },
        "deciles": {
            f"D{i + 1}": _stratum_report(
                decile_labels == i, arrays,
                replicates=args.replicates, confidence=args.confidence, seed=args.seed + 100 + i,
            )
            for i in range(10)
        },
    }

    component_size = arrays["component_size_full_release"].astype(np.int64)
    size_bins = {
        "size_1": component_size == 1,
        "size_2": component_size == 2,
        "size_3_to_5": (component_size >= 3) & (component_size <= 5),
        "size_6_to_10": (component_size >= 6) & (component_size <= 10),
        "size_11_plus": component_size >= 11,
    }
    length_key = "protein_length_residues" if "protein_length_residues" in arrays else "protein_length"
    length = arrays[length_key].astype(np.float64)
    length_labels, length_edges = _quantile_strata(length, 4)
    report["component_size_strata"] = {
        label: _stratum_report(mask, arrays, replicates=args.replicates, confidence=args.confidence, seed=args.seed + 200 + i)
        for i, (label, mask) in enumerate(size_bins.items())
    }
    report["protein_length_strata"] = {
        "quartile_edges": length_edges,
        "quartiles": {
            f"Q{i + 1}": _stratum_report(
                length_labels == i, arrays,
                replicates=args.replicates, confidence=args.confidence, seed=args.seed + 300 + i,
            )
            for i in range(4)
        },
    }

    weight_rows: list[dict[str, Any]] = []
    all_fit_count = 0
    all_fit_failures = 0
    for manifest_path in sorted(args.runs_root.glob("*/run-manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        run_id = manifest["run_id"]
        deployed = None
        for path, fit in _walk_fits(manifest):
            all_fit_count += 1
            if not bool(fit.get("solver_success", False)):
                all_fit_failures += 1
            if path.endswith("/final_refits/distance_stack_audit/primary_fit"):
                deployed = fit
        if deployed is None:
            continue
        weight_rows.append({"run_id": run_id, **_binding_flags(deployed)})

    weight_rows.sort(key=lambda row: row["run_id"])
    if weight_rows:
        def _summary(key: str) -> dict[str, Any]:
            values = np.asarray([row[key] for row in weight_rows if math.isfinite(float(row[key]))], dtype=np.float64)
            if values.size == 0:
                return {"n": 0}
            return {
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "q025": float(np.quantile(values, 0.025)),
                "q975": float(np.quantile(values, 0.975)),
            }
        report["distance_stack_weights"] = {
            "n_jobs": len(weight_rows),
            "all_fits_examined": all_fit_count,
            "all_fit_failures": all_fit_failures,
            "deployed_fit_failures": sum(1 for row in weight_rows if not row["solver_success"]),
            "lambda_counts": {str(value): int(sum(1 for row in weight_rows if row["lambda"] == value)) for value in sorted({row["lambda"] for row in weight_rows})},
            "endpoint_weight_at_zero_jobs": sum(1 for row in weight_rows if row["endpoint_weight_at_zero"]),
            "global_slope_at_zero_jobs": sum(1 for row in weight_rows if row["global_slope_at_zero"]),
            "knn_slope_at_zero_jobs": sum(1 for row in weight_rows if row["knn_slope_at_zero"]),
            "ridge_slope_at_zero_jobs": sum(1 for row in weight_rows if row["ridge_slope_at_zero"]),
            "parameters": {key: _summary(key) for key in (
                "a_global", "a_knn", "a_ridge", "b_global", "b_knn", "b_ridge",
                "solver_iterations", "objective_value",
            )},
            "deployed_rows": weight_rows,
        }

    report["baseline_selection"] = campaign_metrics["primary_analysis"]["job_baseline_selection"]

    (args.outdir / "diagnostics.json").write_text(
        json.dumps(_json_safe(report), indent=1, sort_keys=True, allow_nan=False), encoding="utf-8"
    )

    def _write_tsv(path: Path, header: list[str], rows: Iterable[list[Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("\t".join(header) + "\n")
            for row in rows:
                handle.write("\t".join(str(value) for value in row) + "\n")

    distance_rows = []
    for regime, labels, count in (("quartile", quartile_labels, 4), ("decile", decile_labels, 10)):
        for i in range(count):
            entry = report["distance_strata"]["quartiles" if regime == "quartile" else "deciles"][
                f"{'Q' if regime == 'quartile' else 'D'}{i + 1}"
            ]
            if entry.get("status"):
                continue
            distance_rows.append([
                regime, i + 1, entry["n_genes"], entry["n_components"],
                entry["method"]["r2"], entry["selected_baseline"]["r2"],
                entry["delta_r2_point"],
                entry["bootstrap"]["delta_r2_method_minus_baseline"]["ci_low"],
                entry["bootstrap"]["delta_r2_method_minus_baseline"]["ci_high"],
                entry["method"]["pearson"], entry["selected_baseline"]["pearson"],
                entry["method"]["calibration_slope"], entry["selected_baseline"]["calibration_slope"],
            ])
    _write_tsv(
        args.outdir / "distance_strata.tsv",
        ["regime", "stratum", "n_genes", "n_components", "method_r2", "baseline_r2",
         "delta_r2", "delta_ci_low", "delta_ci_high", "method_pearson",
         "baseline_pearson", "method_calibration_slope", "baseline_calibration_slope"],
        distance_rows,
    )
    _write_tsv(
        args.outdir / "weights_per_job.tsv",
        ["run_id", "a_global", "a_knn", "a_ridge", "b_global", "b_knn", "b_ridge",
         "lambda", "solver_success", "solver_iterations", "endpoint_weight_at_zero",
         "global_slope_at_zero", "knn_slope_at_zero", "ridge_slope_at_zero",
         "sum_a_minus_one", "sum_b"],
        [
            [row[key] for key in (
                "run_id", "a_global", "a_knn", "a_ridge", "b_global", "b_knn", "b_ridge",
                "lambda", "solver_success", "solver_iterations", "endpoint_weight_at_zero",
                "global_slope_at_zero", "knn_slope_at_zero", "ridge_slope_at_zero",
                "sum_a_minus_one", "sum_b",
            )]
            for row in weight_rows
        ],
    )
    print(json.dumps({
        "outdir": str(args.outdir),
        "headline_method_r2": report["headline"]["method"]["r2"],
        "headline_baseline_r2": report["headline"]["selected_baseline"]["r2"],
        "headline_delta_r2": report["headline"]["delta_r2_point"],
        "locked_reference": report.get("locked_reference"),
        "n_jobs": len(weight_rows),
        "solver_failures_all_fits": all_fit_failures,
    }, indent=1))


if __name__ == "__main__":
    main()
