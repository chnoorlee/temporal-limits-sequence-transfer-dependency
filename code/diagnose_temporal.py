#!/usr/bin/env python3
"""Diagnostic recomputation of the frozen 26Q3 time-out analysis.

Uses the diagnostic re-run consensus and the official DepMap 26Q3 file. This
reproduces the temporal scoring primitives but is labelled diagnostic: it does
not replace the hash-bound temporal validator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from diagnose_family_transfer import (  # noqa: E402
    BASELINE_KEY,
    GLOBAL_KEY,
    METHOD_KEY,
    _bootstrap_delta,
    _metrics,
    _quantile_strata,
    _stratum_report,
)
from temporal_26q3_validation import (  # noqa: E402
    align_temporal_cohort,
    parse_gene_headers,
    select_new_models,
)


def _load_26q3(path: Path, header_pattern: str):
    import pandas as pd
    frame = pd.read_csv(path, index_col=0, low_memory=False)
    model_ids = np.asarray([str(value) for value in frame.index], dtype=str)
    symbols, entrez, positions = parse_gene_headers(list(frame.columns), header_pattern)
    values = frame.to_numpy(dtype=np.float32, copy=False)[:, positions]
    values = np.where(np.isfinite(values), values, np.nan).astype(np.float64)
    return model_ids, symbols, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-npz", type=Path, required=True)
    parser.add_argument("--q3-csv", type=Path, required=True)
    parser.add_argument("--models-24q4", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--minimum-new-models", type=int, default=100)
    parser.add_argument("--minimum-finite-fraction", type=float, default=0.8)
    parser.add_argument("--minimum-eligible-genes", type=int, default=5000)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    with np.load(args.consensus_npz, allow_pickle=False) as handle:
        consensus = {key: handle[key] for key in handle.files}
    models_24q4 = [line.strip() for line in args.models_24q4.read_text().splitlines() if line.strip()]

    model_ids, symbols, outcomes = _load_26q3(
        args.q3_csv, r"^(?P<symbol>.+?) \((?P<entrez>\d+)\)$"
    )
    new_models, new_positions = select_new_models(
        model_ids, models_24q4, minimum_new_models=args.minimum_new_models
    )
    cohort = align_temporal_cohort(
        outcome_model_by_gene=outcomes[new_positions],
        outcome_gene_symbols=symbols,
        new_model_ids=new_models,
        consensus_gene_ids=consensus["gene_ids"],
        consensus_canonical_gene_indices=consensus["canonical_gene_indices"],
        consensus_component_ids=consensus["component_ids"],
        primary_prediction=consensus[METHOD_KEY],
        selected_baseline_prediction=consensus[BASELINE_KEY],
        global_prediction=consensus[GLOBAL_KEY],
        minimum_finite_fraction=args.minimum_finite_fraction,
        minimum_eligible_genes=args.minimum_eligible_genes,
    )

    target = cohort.observed_gene_mean
    method = cohort.primary_prediction
    baseline = cohort.selected_baseline_prediction
    global_prediction = cohort.global_prediction
    components = cohort.component_ids.astype(str)

    report: dict[str, Any] = {
        "status": "diagnostic_complete",
        "n_new_models": int(cohort.new_model_ids.size),
        "n_genes": int(target.size),
        "minimum_finite_fraction": args.minimum_finite_fraction,
        "method": _metrics(method, target, global_prediction),
        "selected_baseline": _metrics(baseline, target, global_prediction),
        "bootstrap": _bootstrap_delta(
            method, baseline, global_prediction, target, components,
            replicates=args.replicates, confidence=args.confidence, seed=args.seed,
        ),
    }
    report["delta_r2_point"] = report["method"]["r2"] - report["selected_baseline"]["r2"]

    distance = consensus["consensus_nearest_training_cosine_distance"].astype(np.float64)
    # Map consensus distances through canonical indices.
    distance_by_index = np.full(int(consensus["canonical_gene_indices"].max()) + 1, np.nan)
    distance_by_index[consensus["canonical_gene_indices"].astype(np.int64)] = distance
    cohort_distance = distance_by_index[cohort.canonical_gene_indices.astype(np.int64)]
    labels, edges = _quantile_strata(cohort_distance, 4)
    arrays = {
        "observed_gene_mean": target,
        METHOD_KEY: method,
        BASELINE_KEY: baseline,
        GLOBAL_KEY: global_prediction,
        "component_ids": components,
    }
    report["distance_quartiles"] = {
        "edges": edges,
        "quartiles": {
            f"Q{i + 1}": _stratum_report(
                labels == i, arrays,
                replicates=args.replicates, confidence=args.confidence, seed=args.seed + i,
            )
            for i in range(4)
        },
    }

    component_sizes = np.asarray([
        int(np.sum(components == value)) for value in components
    ], dtype=np.int64)
    size_bins = {
        "size_1": component_sizes == 1,
        "size_2": component_sizes == 2,
        "size_3_to_5": (component_sizes >= 3) & (component_sizes <= 5),
        "size_6_to_10": (component_sizes >= 6) & (component_sizes <= 10),
        "size_11_plus": component_sizes >= 11,
    }
    report["component_size_strata"] = {
        label: _stratum_report(mask, arrays, replicates=args.replicates, confidence=args.confidence, seed=args.seed + 200 + i)
        for i, (label, mask) in enumerate(size_bins.items())
    }

    residual = method - target
    order = np.argsort(residual)
    rows = []
    for position in range(target.size):
        rows.append([
            cohort.gene_ids[position], components[position], int(component_sizes[position]),
            float(cohort_distance[position]), float(target[position]), float(method[position]),
            float(baseline[position]), float(global_prediction[position]), float(residual[position]),
        ])
    with (args.outdir / "temporal_gene_residuals.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("gene\tcomponent_id\tcomponent_size\tconsensus_distance\tobserved_mean\tmethod_prediction\tbaseline_prediction\tglobal_prediction\tresidual\n")
        for row in rows:
            handle.write("\t".join(str(value) for value in row) + "\n")
    report["most_overpredicted"] = [
        {"gene": cohort.gene_ids[i], "residual": float(residual[i])}
        for i in order[-25:][::-1]
    ]
    report["most_underpredicted"] = [
        {"gene": cohort.gene_ids[i], "residual": float(residual[i])}
        for i in order[:25]
    ]
    (args.outdir / "temporal_diagnostics.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, default=lambda value: float(value) if isinstance(value, np.floating) else value),
        encoding="utf-8",
    )
    print(json.dumps({
        "n_new_models": report["n_new_models"],
        "n_genes": report["n_genes"],
        "method_r2": report["method"]["r2"],
        "baseline_r2": report["selected_baseline"]["r2"],
        "delta_r2": report["delta_r2_point"],
        "delta_ci": report["bootstrap"]["delta_r2_method_minus_baseline"],
    }, indent=1))


if __name__ == "__main__":
    main()
