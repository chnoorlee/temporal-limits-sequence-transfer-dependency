#!/usr/bin/env python3
"""Build compact per-gene release artifacts from diagnostic campaign runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


KEYS = {
    "gene_ids": "outer_test_gene_ids",
    "component_ids": "outer_test_component_ids",
    "component_sizes": "outer_test_component_size_full_release",
    "protein_lengths": "outer_test_protein_length_residues",
    "observed_gene_mean": "observed_gene_mean",
    "method_prediction": "family_distance_stack_gene_mean_prediction",
    "baseline_prediction": "job_selected_baseline_gene_mean_prediction",
    "global_prediction": "global_train_mean_prediction",
    "distance": "nearest_outer_train_cosine_distance",
    "weight_global": "distance_stack_weight_global",
    "weight_knn": "distance_stack_weight_knn",
    "weight_ridge": "distance_stack_weight_ridge",
    "selected_baseline": "job_selected_baseline_name",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(path.parent for path in args.runs_root.glob("*/predictions.npz"))
    payload: dict[str, np.ndarray] = {
        "run_ids": np.asarray([path.name for path in run_dirs], dtype="<U64"),
    }
    consensus_rows: dict[str, list[np.ndarray]] = {}
    for run_dir in run_dirs:
        run_id = run_dir.name
        with np.load(run_dir / "predictions.npz", allow_pickle=False) as handle:
            for short, key in KEYS.items():
                payload[f"{short}__{run_id}"] = handle[key]
        manifest = json.loads((run_dir / "run-manifest.json").read_text())
        payload[f"fold__{run_id}"] = np.asarray([manifest["arguments"]["outer_fold"]], dtype=np.int64)
        payload[f"split_seed__{run_id}"] = np.asarray([manifest["arguments"]["split_seed"]], dtype=np.int64)
        payload[f"training_seed__{run_id}"] = np.asarray([manifest["arguments"]["training_seed"]], dtype=np.int64)

    with np.load(args.campaign_dir / "consensus-predictions.npz", allow_pickle=False) as handle:
        consensus = {key: handle[key] for key in handle.files}
    with np.load(args.campaign_dir / "consensus-evidence.npz", allow_pickle=False) as handle:
        evidence_keys = [
            "observed_gene_mean", "component_size_full_release", "protein_length_residues",
            "consensus_nearest_training_cosine_distance", "distance_quartile_zero_based",
            "protein_length_quartile_zero_based", "consensus_distance_stack_weight_global",
            "consensus_distance_stack_weight_knn", "consensus_distance_stack_weight_ridge",
        ]
        evidence = {key: handle[key] for key in evidence_keys}
    genes = consensus["gene_ids"]
    with (args.outdir / "per_gene_consensus.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "gene\tcomponent_id\tcomponent_size\tprotein_length\tconsensus_distance\t"
            "distance_quartile\tprotein_length_quartile\tmethod_prediction\tbaseline_prediction\t"
            "global_prediction\tobserved_gene_mean\tweight_global\tweight_knn\tweight_ridge\n"
        )
        for index in range(genes.size):
            handle.write("\t".join(str(value) for value in (
                genes[index], consensus["component_ids"][index],
                int(evidence["component_size_full_release"][index]),
                int(evidence["protein_length_residues"][index]),
                float(evidence["consensus_nearest_training_cosine_distance"][index]),
                int(evidence["distance_quartile_zero_based"][index]) + 1,
                int(evidence["protein_length_quartile_zero_based"][index]) + 1,
                float(consensus["family_distance_stack_gene_mean_prediction"][index]),
                float(consensus["consensus_per_job_selected_baseline_gene_mean_prediction"][index]),
                float(consensus["consensus_global_train_mean_prediction"][index]),
                float(evidence["observed_gene_mean"][index]),
                float(evidence["consensus_distance_stack_weight_global"][index]),
                float(evidence["consensus_distance_stack_weight_knn"][index]),
                float(evidence["consensus_distance_stack_weight_ridge"][index]),
            )) + "\n")

    archive = args.outdir / "per_job_gene_predictions.npz"
    np.savez_compressed(archive, **payload)
    records = {
        "per_job_gene_predictions.npz": {
            "bytes": archive.stat().st_size, "sha256": sha256(archive),
            "description": "Per-job outer-test gene-mean predictions, component IDs, distances and learned weights.",
        },
        "per_gene_consensus.tsv": {
            "bytes": (args.outdir / "per_gene_consensus.tsv").stat().st_size,
            "sha256": sha256(args.outdir / "per_gene_consensus.tsv"),
            "description": "Consensus per-gene predictions and component annotations for 17,768 genes.",
        },
    }
    (args.outdir / "release_manifest.json").write_text(
        json.dumps(records, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
