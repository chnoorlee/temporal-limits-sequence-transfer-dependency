#!/usr/bin/env python3
"""Independently verify a completed family-holdout model campaign.

The verifier deliberately reimplements scoring instead of importing the runner's
metric code. It binds predictions to source indices and targets, refits the
training-gene-only references, and proves exact-once outer prediction coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from embedding_provenance import validate_embedding_bundle


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "DepPLM-family-holdout-model-evaluation-v2"
RELEASE_SCHEMA = "DepPLM-family-holdout-model-release-v2"
AGGREGATE_SCHEMA = "DepPLM-family-holdout-aggregate-v2"
N_FOLDS = 5
REGISTERED_SPLIT_SEEDS = (42, 43, 44)
REGISTERED_TRAINING_SEEDS = (42, 43, 44)
NEURAL_CONDITIONS = ("DepPLM", "C-PLM-MLP")
BASELINE_CONDITION = "TRAIN-GENE-CELL-MEAN"
ALL_CONDITIONS = (*NEURAL_CONDITIONS, BASELINE_CONDITION)
CANONICAL_ARGUMENTS = {
    "inner_max_epochs": 20,
    "patience": 5,
    "batch_size": 64,
    "gene_subsample": 6_000,
    "top_mut": 5_000,
    "n_expr_pc": 128,
}
CANONICAL_PASSIVE_ARGUMENTS = {"pilot_cells": 64, "pilot_genes": 250}
REGISTERED_MANIFEST = (
    ROOT / "results" / "family_holdout_v1" / "identity_30" / "fold_manifest.tsv"
)
EXPECTED_SOURCE_PATHS = (
    "code/run_family_holdout_experiment.py",
    "code/family_holdout.py",
    "code/verify_family_holdout_experiment.py",
    "code/run_corrected_experiments.py",
    "code/corrected_protocol.py",
    "code/models.py",
    "code/03_embed_proteins.py",
    "code/embedding_provenance.py",
    "data/processed/gene_effect.npy",
    "data/processed/esm2_t6_8M_residue_chunked.npy",
    "data/processed/esm2_t6_8M_residue_chunked.l2.npy",
    "data/processed/esm2_t6_8M_residue_chunked.names.txt",
    "data/processed/esm2_t6_8M_residue_chunked.manifest.json",
    "data/processed/gene_seqs.fa",
    "data/processed/ppi_graph.npz",
    "data/processed/genes.txt",
    "data/processed/cell_lines.txt",
    "data/processed/lineages.txt",
    "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "data/depmap/OmicsSomaticMutations.csv",
    "results/family_holdout_v1/release_manifest.json",
    "results/family_holdout_v1/identity_30/fold_manifest.tsv",
    "results/family_holdout_v1/identity_30/split_summary.json",
)


class VerificationError(RuntimeError):
    """Raised when a persisted campaign fails an independent check."""


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def correlations(
    prediction: np.ndarray, target: np.ndarray, *, axis: int, rank: bool = False
) -> np.ndarray:
    n_items = prediction.shape[0] if axis == 1 else prediction.shape[1]
    output = np.full(n_items, np.nan, dtype=np.float64)
    for index in range(n_items):
        x = prediction[index] if axis == 1 else prediction[:, index]
        y = target[index] if axis == 1 else target[:, index]
        finite = np.isfinite(x) & np.isfinite(y)
        if int(finite.sum()) < 3:
            continue
        x = np.asarray(x[finite], dtype=np.float64)
        y = np.asarray(y[finite], dtype=np.float64)
        if rank:
            x, y = rankdata(x), rankdata(y)
        if np.std(x) > 0.0 and np.std(y) > 0.0:
            output[index] = float(np.corrcoef(x, y)[0, 1])
    return output


def finite_summary(values: np.ndarray, prefix: str) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
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


def independent_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    cell_reference: np.ndarray,
    global_reference: float | np.ndarray,
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float32)
    observed = np.asarray(target, dtype=np.float32)
    reference = np.asarray(cell_reference, dtype=np.float32)
    if reference.ndim == 1:
        reference = np.broadcast_to(reference[:, None], predicted.shape)
    global_values = np.asarray(global_reference, dtype=np.float32)
    if global_values.ndim == 0:
        global_values = np.broadcast_to(global_values, predicted.shape)
        global_value: float | None = float(global_reference)
        global_scope = "single_outer_training_mean"
    else:
        global_value = None
        global_scope = "fold_specific_outer_training_mean_matrix"
    if not (
        predicted.shape == observed.shape == reference.shape == global_values.shape
    ):
        raise VerificationError("metric inputs have incompatible shapes")
    valid = np.isfinite(observed) & np.isfinite(reference)
    if not np.isfinite(predicted[valid]).all() or not np.isfinite(global_values[valid]).all():
        raise VerificationError("non-finite evaluable prediction or reference")
    n_pairs = int(valid.sum())
    if n_pairs == 0:
        raise VerificationError("no finite target pairs")
    error = np.where(valid, predicted - observed, 0.0)
    cell_error = np.where(valid, reference - observed, 0.0)
    global_error = np.where(valid, global_values - observed, 0.0)
    sse = float(np.square(error, dtype=np.float64).sum(dtype=np.float64))
    cell_sse = float(np.square(cell_error, dtype=np.float64).sum(dtype=np.float64))
    global_sse = float(np.square(global_error, dtype=np.float64).sum(dtype=np.float64))
    result: dict[str, Any] = {
        "mse": sse / n_pairs,
        "rmse": math.sqrt(sse / n_pairs),
        "mae": float(np.abs(error).sum(dtype=np.float64)) / n_pairs,
        "train_gene_set_cell_mean_mse": cell_sse / n_pairs,
        "train_gene_set_cell_mean_mae": float(np.abs(cell_error).sum(dtype=np.float64)) / n_pairs,
        "residual_r2_vs_train_gene_set_cell_mean": 1.0 - sse / cell_sse if cell_sse > 0 else None,
        "global_train_mean": global_value,
        "global_train_reference_scope": global_scope,
        "global_train_mean_mse": global_sse / n_pairs,
        "residual_r2_vs_global_train_mean": 1.0 - sse / global_sse if global_sse > 0 else None,
        "n_finite_pairs": n_pairs,
        "n_cells": int(predicted.shape[0]),
        "n_held_out_genes": int(predicted.shape[1]),
    }
    result.update(finite_summary(correlations(predicted, observed, axis=1), "cellwise_pearson"))
    result.update(finite_summary(correlations(predicted, observed, axis=0), "gene_wise_pearson"))
    result.update(
        finite_summary(
            correlations(predicted - reference, observed - reference, axis=1),
            "residual_cellwise_pearson",
        )
    )
    mask = valid.reshape(-1)
    x, y = predicted.reshape(-1)[mask], observed.reshape(-1)[mask]
    result["pooled_pearson"] = (
        float(np.corrcoef(x, y)[0, 1])
        if np.std(x) > 0.0 and np.std(y) > 0.0
        else None
    )
    result.update(
        finite_summary(
            correlations(predicted, observed, axis=1, rank=True),
            "cellwise_spearman",
        )
    )
    return result


def assert_metric_mapping(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, context: str
) -> None:
    if set(observed) != set(expected):
        raise VerificationError(f"{context}: metric keys differ")
    for key, expected_value in expected.items():
        observed_value = observed[key]
        if expected_value is None or isinstance(expected_value, str):
            if observed_value != expected_value:
                raise VerificationError(f"{context}: {key} differs")
        elif isinstance(expected_value, int):
            if observed_value != expected_value:
                raise VerificationError(f"{context}: {key} differs")
        elif not math.isclose(
            float(observed_value), float(expected_value), rel_tol=2e-7, abs_tol=2e-8
        ):
            raise VerificationError(
                f"{context}: {key} differs ({observed_value} != {expected_value})"
            )


def verify_release_inventory(campaign: Path) -> dict[str, Mapping[str, Any]]:
    release_path = campaign / "release-manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema") != RELEASE_SCHEMA or release.get("status") != "complete":
        raise VerificationError("release manifest is not complete")
    records = {str(item["path"]): item for item in release.get("files", [])}
    if len(records) != len(release.get("files", [])):
        raise VerificationError("duplicate release paths")
    actual = {
        path.relative_to(campaign).as_posix()
        for path in campaign.rglob("*")
        if path.is_file()
        and path.name != "release-manifest.json"
        and not any(part.startswith(".") for part in path.relative_to(campaign).parts)
    }
    if actual != set(records):
        raise VerificationError("release inventory does not exactly match campaign files")
    for relative, record in records.items():
        path = campaign / relative
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise VerificationError(f"release file binding failed: {relative}")
    return records


def _registered_rows(path: Path, genes: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["gene", "component_id", "fold", "stratum"]:
            raise VerificationError("registered fold-manifest columns differ")
        rows = [dict(row) for row in reader]
    if [row["gene"] for row in rows] != list(genes):
        raise VerificationError("registered fold-manifest gene order differs")
    if len({row["gene"] for row in rows}) != len(rows):
        raise VerificationError("registered fold-manifest has duplicate genes")
    return rows


def reconstruct_seeded_manifest(
    path: Path,
    genes: Sequence[str],
    *,
    split_seed: int,
) -> tuple[bytes, list[dict[str, Any]]]:
    """Independently replay component assignment without importing the runner."""
    rows = _registered_rows(path, genes)
    members: dict[str, list[str]] = {}
    for row in rows:
        members.setdefault(row["component_id"], []).append(row["gene"])

    def tie_hash(component_id: str) -> str:
        return hashlib.sha256(
            f"{int(split_seed)}\0{component_id}".encode("utf-8")
        ).hexdigest()

    fold_sizes = [0] * N_FOLDS
    by_gene: dict[str, int] = {}
    ordered = sorted(
        members.items(),
        key=lambda item: (-len(item[1]), tie_hash(item[0]), item[0]),
    )
    for component_id, component_genes in ordered:
        fold = min(range(N_FOLDS), key=lambda index: (fold_sizes[index], index))
        fold_sizes[fold] += len(component_genes)
        for gene in component_genes:
            if gene in by_gene:
                raise VerificationError(f"gene assigned twice while replaying split: {gene}")
            by_gene[gene] = fold
    if set(by_gene) != set(genes) or any(size == 0 for size in fold_sizes):
        raise VerificationError("replayed split is incomplete")
    reconstructed = [
        {
            "gene": row["gene"],
            "component_id": row["component_id"],
            "fold": int(by_gene[row["gene"]]),
            "stratum": row["stratum"],
        }
        for row in rows
    ]
    lines = ["gene\tcomponent_id\tfold\tstratum"]
    lines.extend(
        f"{row['gene']}\t{row['component_id']}\t{row['fold']}\t{row['stratum']}"
        for row in reconstructed
    )
    return ("\n".join(lines) + "\n").encode("utf-8"), reconstructed


def validate_completion_contract(
    run: Mapping[str, Any], completion: Mapping[str, Any]
) -> None:
    if run.get("schema") != SCHEMA or completion.get("schema") != SCHEMA:
        raise VerificationError("campaign schema mismatch")
    configuration = run.get("configuration")
    if not isinstance(configuration, Mapping):
        raise VerificationError("run configuration is missing")
    split_seed = configuration.get("split_seed")
    training_seed = configuration.get("training_seed")
    if split_seed not in REGISTERED_SPLIT_SEEDS:
        raise VerificationError("split_seed is outside the registered campaign")
    if training_seed not in REGISTERED_TRAINING_SEEDS:
        raise VerificationError("training_seed is outside the registered campaign")
    if configuration.get("fold_id") != -1:
        raise VerificationError("registered campaign cell must contain all folds")
    if configuration.get("conditions") != list(NEURAL_CONDITIONS):
        raise VerificationError("registered campaign cell has wrong neural conditions")
    if configuration.get("pilot") is not False:
        raise VerificationError("pilot output cannot satisfy the registered campaign")
    for key, expected in CANONICAL_ARGUMENTS.items():
        if configuration.get(key) != expected:
            raise VerificationError(f"noncanonical campaign argument: {key}")
    for key, expected in CANONICAL_PASSIVE_ARGUMENTS.items():
        if configuration.get(key) != expected:
            raise VerificationError(f"noncanonical campaign argument: {key}")
    expected_configuration_keys = {
        "split_seed",
        "training_seed",
        "fold_id",
        "conditions",
        "pilot",
        *CANONICAL_ARGUMENTS,
        *CANONICAL_PASSIVE_ARGUMENTS,
    }
    if set(configuration) != expected_configuration_keys:
        raise VerificationError("run configuration field inventory differs")
    if run.get("canonical_deviations") != []:
        raise VerificationError("run manifest records canonical deviations")
    expected = {
        "status": "campaign_cell_complete",
        "scientific_result": False,
        "campaign_aggregation_required": True,
        "registered_campaign_cell_eligible": True,
        "canonical_deviations": [],
        "completed_folds": list(range(N_FOLDS)),
        "completed_conditions": list(ALL_CONDITIONS),
        "source_stability": "verified",
        "holdout_axis": "genes",
        "cell_holdout": False,
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise VerificationError(f"completion contract differs: {key}")


def validate_array_schema(
    observed: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
    *,
    context: str,
) -> None:
    if set(observed) != set(expected):
        raise VerificationError(f"{context}: prediction array inventory differs")
    for name, expected_value in expected.items():
        value = observed[name]
        if value.dtype != expected_value.dtype:
            raise VerificationError(
                f"{context}: {name} dtype differs ({value.dtype} != {expected_value.dtype})"
            )
        if value.shape != expected_value.shape:
            raise VerificationError(
                f"{context}: {name} shape differs ({value.shape} != {expected_value.shape})"
            )


def fold_descriptive_summary(values: Sequence[float | None]) -> dict[str, Any]:
    normalized = [
        None if value is None or not math.isfinite(float(value)) else float(value)
        for value in values
    ]
    finite = [value for value in normalized if value is not None]
    if len(finite) != len(normalized) or len(finite) < 2:
        return {
            "status": "undefined_nonfinite_fold_metric",
            "n_folds": len(normalized),
            "n_finite_folds": len(finite),
            "mean": None,
            "sd": None,
            "minimum": None,
            "maximum": None,
            "fold_values": normalized,
        }
    array = np.asarray(finite, dtype=np.float64)
    return {
        "status": "descriptive_fold_dispersion",
        "n_folds": int(array.size),
        "n_finite_folds": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "fold_values": array.tolist(),
    }


def assert_nested_mapping(observed: Any, expected: Any, *, context: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise VerificationError(f"{context}: mapping keys differ")
        for key, value in expected.items():
            assert_nested_mapping(observed[key], value, context=f"{context}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise VerificationError(f"{context}: list length differs")
        for index, value in enumerate(expected):
            assert_nested_mapping(
                observed[index], value, context=f"{context}[{index}]"
            )
        return
    if isinstance(expected, float):
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(observed), expected, rel_tol=2e-7, abs_tol=2e-8
        ):
            raise VerificationError(f"{context}: numeric value differs")
        return
    if observed != expected:
        raise VerificationError(f"{context}: value differs")


def _expected_split_record(
    assignments: Sequence[Mapping[str, Any]],
    genes: Sequence[str],
    cells: Sequence[str],
    outer_fold: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    folds = np.asarray([int(item["fold"]) for item in assignments], dtype=np.int64)
    inner_fold = (outer_fold + 1) % N_FOLDS
    indices = {
        "outer_train": np.flatnonzero(folds != outer_fold).astype(np.int64),
        "inner_train": np.flatnonzero(
            (folds != outer_fold) & (folds != inner_fold)
        ).astype(np.int64),
        "inner_validation": np.flatnonzero(folds == inner_fold).astype(np.int64),
        "outer_test": np.flatnonzero(folds == outer_fold).astype(np.int64),
    }

    def block(values: np.ndarray) -> dict[str, Any]:
        components = sorted(
            {str(assignments[int(index)]["component_id"]) for index in values}
        )
        return {
            "n_genes": int(values.size),
            "n_components": len(components),
            "gene_indices": values.tolist(),
            "source_gene_indices": values.tolist(),
            "gene_indices_sha256": hash_array(values.astype("<i8")),
            "gene_ids_sha256": hash_strings([genes[int(index)] for index in values]),
            "component_ids_sha256": hash_strings(components),
        }

    record = {
        "schema": "DepPLM-family-holdout-nested-split-v1",
        "holdout_axis": "genes",
        "cell_holdout": False,
        "outer_fold": outer_fold,
        "inner_validation_fold": inner_fold,
        "main_identity_percentage": 30.0,
        "minimum_bidirectional_coverage": 0.8,
        **{name: block(values) for name, values in indices.items()},
        "n_evaluation_cells": len(cells),
        "evaluation_cell_ids_sha256": hash_strings(cells),
        "label_isolation": (
            "outer-test gene-effect labels are accessed only by the one final "
            "evaluation after tuning and fresh outer-train refit"
        ),
    }
    return record, indices


def validate_run_provenance(run: Mapping[str, Any]) -> None:
    expected_protocol = {
        "holdout_axis": "genes",
        "cell_holdout": False,
        "outer_folds": N_FOLDS,
        "inner_validation_rule": "(outer_fold + 1) mod 5",
        "main_identity_percentage": 30.0,
        "minimum_bidirectional_coverage": 0.8,
        "feature_access": "label-inductive, feature-transductive",
        "outer_test_access": "one call after selection and fresh refit",
    }
    if run.get("status") != "initialized" or run.get("protocol") != expected_protocol:
        raise VerificationError("run protocol provenance differs")

    device = run.get("device")
    if device not in ("cpu", "cuda"):
        raise VerificationError("run device provenance differs")

    environment = run.get("environment")
    expected_environment_keys = {
        "python",
        "python_executable",
        "platform",
        "machine",
        "numpy",
        "pandas",
        "scipy",
        "scikit_learn",
        "torch",
        "torch_cuda",
        "cudnn",
        "cuda_available",
        "cuda_device_name",
        "deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "installed_distributions",
        "installed_distributions_sha256",
    }
    if not isinstance(environment, Mapping) or set(environment) != expected_environment_keys:
        raise VerificationError("run environment field inventory differs")
    if device == "cuda" and environment.get("cuda_available") is not True:
        raise VerificationError("CUDA run was recorded without CUDA availability")
    if (
        environment.get("deterministic_algorithms") is not True
        or environment.get("cudnn_deterministic") is not True
        or environment.get("cudnn_benchmark") is not False
    ):
        raise VerificationError("run environment does not record deterministic training")
    distributions = environment.get("installed_distributions")
    if not isinstance(distributions, list) or not distributions:
        raise VerificationError("run installed-distribution inventory is missing")
    normalized_names: list[str] = []
    for record in distributions:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"name", "version"}
            or not isinstance(record.get("name"), str)
            or not record["name"]
            or not isinstance(record.get("version"), str)
            or not record["version"]
        ):
            raise VerificationError("run installed-distribution record differs")
        normalized_names.append(record["name"].casefold())
    if normalized_names != sorted(normalized_names) or len(normalized_names) != len(
        set(normalized_names)
    ):
        raise VerificationError("run installed-distribution inventory is not unique/sorted")
    expected_environment_hash = canonical_json_sha256(
        {"installed_distributions": distributions}
    )
    if environment.get("installed_distributions_sha256") != expected_environment_hash:
        raise VerificationError("run installed-distribution hash differs")

    git = run.get("git")
    if not isinstance(git, Mapping) or set(git) != {"commit", "dirty"}:
        raise VerificationError("run git provenance differs")
    if git["commit"] is not None and (
        not isinstance(git["commit"], str) or len(git["commit"]) != 40
    ):
        raise VerificationError("run git commit is invalid")
    if git["dirty"] not in (True, False, None):
        raise VerificationError("run git dirty flag is invalid")

    expected_embedding = validate_embedding_bundle(ROOT)
    if run.get("esm_embedding") != expected_embedding:
        raise VerificationError("verified ESM embedding provenance differs")

    family_release_path = ROOT / "results" / "family_holdout_v1" / "release_manifest.json"
    family_release = json.loads(family_release_path.read_text(encoding="utf-8"))
    release_records = {
        str(record["path"]): record for record in family_release.get("files", [])
    }
    required_family_paths = (
        "identity_30/fold_manifest.tsv",
        "identity_30/split_summary.json",
        "homology_hits.tsv.gz",
    )
    if any(path not in release_records for path in required_family_paths):
        raise VerificationError("registered family release lacks required artifacts")
    expected_family = {
        "verification_status": "verified",
        "manifest": {
            "path": family_release_path.relative_to(ROOT).as_posix(),
            "bytes": family_release_path.stat().st_size,
            "sha256": sha256_file(family_release_path),
        },
        "manifest_sha256": sha256_file(family_release_path),
        "main_identity_percentage": 30.0,
        "minimum_bidirectional_coverage": 0.8,
        "fold_manifest": release_records["identity_30/fold_manifest.tsv"],
        "split_summary": release_records["identity_30/split_summary.json"],
        "homology_hits": release_records["homology_hits.tsv.gz"],
        "cross_fold_qualifying_edges": 0,
        "exact_duplicate_fold_violations": 0,
    }
    if run.get("family_release") != expected_family:
        raise VerificationError("registered family-release provenance differs")


def verify_campaign(campaign: Path) -> dict[str, Any]:
    campaign = campaign.resolve(strict=True)
    verify_release_inventory(campaign)
    run = json.loads((campaign / "run-manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((campaign / "completion.json").read_text(encoding="utf-8"))
    validate_completion_contract(run, completion)
    validate_run_provenance(run)
    sources = run.get("sources")
    if not isinstance(sources, list):
        raise VerificationError("source inventory is missing")
    source_paths = [str(source.get("path")) for source in sources]
    if source_paths != list(EXPECTED_SOURCE_PATHS):
        raise VerificationError("source inventory does not match the required exact set/order")
    if len(set(source_paths)) != len(source_paths):
        raise VerificationError("source inventory contains duplicates")
    for source in sources:
        source_path = ROOT / source["path"]
        if not source_path.is_file() or source_path.stat().st_size != int(source["bytes"]):
            raise VerificationError(f"source binding failed: {source['path']}")
        if sha256_file(source_path) != source["sha256"]:
            raise VerificationError(f"source SHA-256 failed: {source['path']}")

    gene_effect = np.load(ROOT / "data" / "processed" / "gene_effect.npy", mmap_mode="r")
    genes = read_lines(ROOT / "data" / "processed" / "genes.txt")
    cells = read_lines(ROOT / "data" / "processed" / "cell_lines.txt")
    if gene_effect.dtype != np.dtype("float32") or gene_effect.shape != (
        len(genes),
        len(cells),
    ):
        raise VerificationError("gene_effect source schema differs")
    configuration = run["configuration"]
    split_seed = int(configuration["split_seed"])
    training_seed = int(configuration["training_seed"])
    expected_manifest_bytes, assignments = reconstruct_seeded_manifest(
        REGISTERED_MANIFEST,
        genes,
        split_seed=split_seed,
    )
    persisted_manifest = campaign / "fold-manifest.tsv"
    if persisted_manifest.read_bytes() != expected_manifest_bytes:
        raise VerificationError("persisted fold manifest differs from independent replay")
    if split_seed == 42 and persisted_manifest.read_bytes() != REGISTERED_MANIFEST.read_bytes():
        raise VerificationError("split seed 42 does not exactly reproduce the registered manifest")
    fold_manifest_record = run.get("fold_manifest")
    if not isinstance(fold_manifest_record, Mapping):
        raise VerificationError("run manifest lacks fold-manifest provenance")
    fold_counts = [
        sum(int(item["fold"]) == fold for item in assignments)
        for fold in range(N_FOLDS)
    ]
    expected_fold_record = {
        "schema": "DepPLM-family-holdout-fold-manifest-v2",
        "split_seed": split_seed,
        "assignment_algorithm": (
            "components_descending_size_then_sha256_seed_component_id; "
            "greedy_smallest_fold_then_lowest_fold_index"
        ),
        "gene_count": len(genes),
        "component_count": len({item["component_id"] for item in assignments}),
        "fold_gene_counts": fold_counts,
        "seed_42_exact_registered_reproduction": True if split_seed == 42 else None,
    }
    for key, value in expected_fold_record.items():
        if fold_manifest_record.get(key) != value:
            raise VerificationError(f"fold-manifest provenance differs: {key}")
    artifact = fold_manifest_record.get("artifact", {})
    if artifact.get("path") != "fold-manifest.tsv":
        raise VerificationError("fold-manifest artifact path differs")
    if int(artifact.get("bytes", -1)) != persisted_manifest.stat().st_size:
        raise VerificationError("fold-manifest byte count differs")
    if artifact.get("sha256") != sha256_file(persisted_manifest):
        raise VerificationError("fold-manifest hash differs")
    registered_source = fold_manifest_record.get("registered_component_source")
    expected_registered_source = {
        "path": REGISTERED_MANIFEST.relative_to(ROOT).as_posix(),
        "bytes": REGISTERED_MANIFEST.stat().st_size,
        "sha256": sha256_file(REGISTERED_MANIFEST),
    }
    if registered_source != expected_registered_source:
        raise VerificationError("registered component-manifest provenance differs")

    folds = list(range(N_FOLDS))
    conditions = list(ALL_CONDITIONS)
    fold_directories = {
        path.name for path in campaign.iterdir() if path.is_dir() and path.name.startswith("fold-")
    }
    if fold_directories != {f"fold-{fold:02d}" for fold in folds}:
        raise VerificationError("campaign does not contain exactly five canonical fold directories")
    prediction_columns: dict[str, dict[int, np.ndarray]] = {name: {} for name in conditions}
    cell_references: dict[str, dict[int, np.ndarray]] = {name: {} for name in conditions}
    global_references: dict[str, dict[int, float]] = {name: {} for name in conditions}
    per_fold_metrics: dict[str, dict[int, dict[str, Any]]] = {
        name: {} for name in conditions
    }
    prediction_records: dict[str, list[dict[str, Any]]] = {
        name: [] for name in conditions
    }
    fold_gene_sets: dict[int, np.ndarray] = {}
    metric_count = 0
    expected_source_cells = np.arange(len(cells), dtype=np.int64)

    for fold in folds:
        fold_dir = campaign / f"fold-{fold:02d}"
        split = json.loads((fold_dir / "split.json").read_text(encoding="utf-8"))
        expected_split, split_indices = _expected_split_record(
            assignments, genes, cells, fold
        )
        if split != expected_split:
            raise VerificationError(
                f"fold {fold}: split.json differs from independently reconstructed outer/inner split"
            )
        test_local = split_indices["outer_test"]
        test_source = test_local
        train_source = split_indices["outer_train"]
        inner_train_local = split_indices["inner_train"]
        inner_validation_local = split_indices["inner_validation"]
        outer_train_local = split_indices["outer_train"]
        if int(split["inner_validation_fold"]) != (fold + 1) % N_FOLDS:
            raise VerificationError(f"fold {fold}: inner validation rule differs")
        if (
            np.intersect1d(test_local, outer_train_local).size
            or np.intersect1d(test_local, inner_train_local).size
            or np.intersect1d(test_local, inner_validation_local).size
            or np.intersect1d(inner_train_local, inner_validation_local).size
        ):
            raise VerificationError(f"fold {fold}: reconstructed split has leakage")
        if not np.array_equal(
            np.sort(np.concatenate([inner_train_local, inner_validation_local])),
            outer_train_local,
        ):
            raise VerificationError(f"fold {fold}: inner split does not cover outer training")
        fold_gene_sets[fold] = test_source
        condition_directories = {
            path.name for path in fold_dir.iterdir() if path.is_dir()
        }
        if condition_directories != set(conditions):
            raise VerificationError(f"fold {fold}: condition directory inventory differs")

        train = np.asarray(
            gene_effect[np.ix_(train_source, expected_source_cells)], dtype=np.float32
        )
        cell_mean = np.nanmean(train, axis=0, dtype=np.float64).astype(np.float32)
        global_mean = float(np.nanmean(train, dtype=np.float64))
        target = np.asarray(
            gene_effect[np.ix_(test_source, expected_source_cells)].T,
            dtype=np.float32,
        )

        expected_test_for_fold: np.ndarray | None = None
        for condition in conditions:
            condition_dir = fold_dir / condition
            result = json.loads((condition_dir / "result.json").read_text(encoding="utf-8"))
            if (
                result.get("schema") != SCHEMA
                or result.get("condition") != condition
                or int(result.get("outer_fold", -1)) != fold
                or result.get("holdout_axis") != "genes"
                or result.get("cell_holdout") is not False
                or result.get("split_seed") != split_seed
                or result.get("training_seed") != training_seed
                or result.get("outer_test_evaluation_calls") != 1
            ):
                raise VerificationError(f"fold {fold} {condition}: result identity differs")
            if result.get("split_sha256") != canonical_json_sha256(split):
                raise VerificationError(f"fold {fold} {condition}: split binding differs")
            prediction_path = condition_dir / "outer-test-predictions.npz"
            prediction_descriptor = result.get("outer_test_predictions", {})
            if prediction_descriptor.get("schema") != "DepPLM-family-holdout-outer-predictions-v2":
                raise VerificationError(f"fold {fold} {condition}: prediction schema differs")
            if prediction_descriptor.get("path") != "outer-test-predictions.npz":
                raise VerificationError(f"fold {fold} {condition}: prediction path differs")
            if int(prediction_descriptor.get("bytes", -1)) != prediction_path.stat().st_size:
                raise VerificationError(f"fold {fold} {condition}: prediction bytes differ")
            if sha256_file(prediction_path) != prediction_descriptor.get("sha256"):
                raise VerificationError(f"fold {fold} {condition}: prediction SHA-256 failed")
            with np.load(prediction_path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            expected_arrays = {
                "prediction": np.empty(target.shape, dtype=np.float32),
                "train_gene_set_cell_mean": cell_mean,
                "global_train_mean": np.asarray([global_mean], dtype=np.float32),
                "outer_test_gene_indices": test_local,
                "outer_train_gene_indices": outer_train_local,
                "inner_train_gene_indices": inner_train_local,
                "inner_validation_gene_indices": inner_validation_local,
                "source_test_gene_indices": test_source,
                "source_cell_indices": expected_source_cells,
                "test_gene_ids": np.asarray(
                    [genes[int(index)] for index in test_source], dtype=str
                ),
                "cell_ids": np.asarray(cells, dtype=str),
            }
            validate_array_schema(
                arrays,
                expected_arrays,
                context=f"fold {fold} {condition}",
            )
            metadata = prediction_descriptor.get("arrays", {})
            if set(arrays) != set(metadata):
                raise VerificationError(f"fold {fold} {condition}: array inventory differs")
            for name, value in arrays.items():
                record = metadata[name]
                if set(record) != {"shape", "dtype", "sha256"}:
                    raise VerificationError(
                        f"fold {fold} {condition}: {name} metadata schema differs"
                    )
                if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
                    raise VerificationError(f"fold {fold} {condition}: {name} metadata differs")
                if hash_array(value) != record["sha256"]:
                    raise VerificationError(f"fold {fold} {condition}: {name} hash differs")
            if not np.array_equal(arrays["outer_test_gene_indices"], test_local):
                raise VerificationError(f"fold {fold} {condition}: local test indices differ")
            for name, expected in (
                ("outer_train_gene_indices", outer_train_local),
                ("inner_train_gene_indices", inner_train_local),
                ("inner_validation_gene_indices", inner_validation_local),
            ):
                if not np.array_equal(arrays[name], expected):
                    raise VerificationError(f"fold {fold} {condition}: {name} differs")
            if not np.array_equal(arrays["source_test_gene_indices"], test_source):
                raise VerificationError(f"fold {fold} {condition}: source test indices differ")
            if expected_test_for_fold is None:
                expected_test_for_fold = arrays["source_test_gene_indices"]
            elif not np.array_equal(expected_test_for_fold, arrays["source_test_gene_indices"]):
                raise VerificationError(f"fold {fold}: conditions cover different genes")
            source_cells = arrays["source_cell_indices"]
            if not np.array_equal(expected_source_cells, source_cells):
                raise VerificationError("conditions use different evaluation cells")
            if arrays["cell_ids"].tolist() != [cells[int(index)] for index in source_cells]:
                raise VerificationError(f"fold {fold} {condition}: cell identifiers differ")
            if arrays["test_gene_ids"].tolist() != [genes[int(index)] for index in test_source]:
                raise VerificationError(f"fold {fold} {condition}: gene identifiers differ")
            expected_target_record = {
                "source": "data/processed/gene_effect.npy",
                "shape": list(target.shape),
                "dtype": "float32",
                "sha256": hash_array(target),
                "persisted_in_prediction_npz": False,
            }
            if result.get("outer_test_target") != expected_target_record:
                raise VerificationError(f"fold {fold} {condition}: target binding differs")
            if not np.array_equal(arrays["train_gene_set_cell_mean"], cell_mean):
                raise VerificationError(f"fold {fold} {condition}: training cell mean differs")
            if not math.isclose(
                float(arrays["global_train_mean"][0]),
                global_mean,
                rel_tol=2e-7,
                abs_tol=2e-8,
            ):
                raise VerificationError(f"fold {fold} {condition}: training global mean differs")
            prediction = arrays["prediction"]
            if not np.isfinite(prediction).all():
                raise VerificationError(f"fold {fold} {condition}: prediction is non-finite")
            recomputed = independent_metrics(prediction, target, cell_mean, global_mean)
            assert_metric_mapping(
                result["test_metrics"], recomputed, context=f"fold {fold} {condition}"
            )
            if condition == "TRAIN-GENE-CELL-MEAN" and not np.array_equal(
                prediction, np.broadcast_to(cell_mean[:, None], prediction.shape)
            ):
                raise VerificationError(f"fold {fold}: baseline predictions differ")
            if condition != BASELINE_CONDITION:
                expected_inner_hash = hash_array(inner_train_local.astype("<i8"))
                tuning = result.get("tuning")
                if not isinstance(tuning, list) or len(tuning) != 4:
                    raise VerificationError(f"fold {fold} {condition}: tuning grid differs")
                for candidate in tuning:
                    training = candidate.get("training", {})
                    if (
                        training.get("training_gene_indices_sha256")
                        != expected_inner_hash
                        or training.get("n_training_genes")
                        != int(inner_train_local.size)
                    ):
                        raise VerificationError(
                            f"fold {fold} {condition}: inner training genes differ"
                        )
                refit = result.get("refit", {})
                if (
                    refit.get("training_gene_indices_sha256")
                    != hash_array(outer_train_local.astype("<i8"))
                    or refit.get("n_training_genes") != int(outer_train_local.size)
                ):
                    raise VerificationError(
                        f"fold {fold} {condition}: outer refit genes differ"
                    )
            prediction_columns[condition][fold] = prediction
            cell_references[condition][fold] = cell_mean
            # The pooled artifact intentionally uses the persisted float32
            # reference, so verify the exact bytes used by aggregation.
            global_references[condition][fold] = float(
                arrays["global_train_mean"][0]
            )
            per_fold_metrics[condition][fold] = dict(result["test_metrics"])
            prediction_records[condition].append(
                {
                    "outer_fold": fold,
                    "path": (
                        f"fold-{fold:02d}/{condition}/outer-test-predictions.npz"
                    ),
                    "sha256": prediction_descriptor["sha256"],
                }
            )
            metric_count += 1

    concatenated = np.concatenate([fold_gene_sets[fold] for fold in folds])
    if np.unique(concatenated).size != concatenated.size:
        raise VerificationError("a held-out gene is predicted more than once")
    expected_source_genes = np.sort(np.concatenate(list(fold_gene_sets.values())))
    if not np.array_equal(expected_source_genes, np.arange(len(genes), dtype=np.int64)):
        raise VerificationError("five-fold campaign does not predict every gene exactly once")

    aggregate_path = campaign / "aggregate-summary.json"
    if not aggregate_path.is_file():
        raise VerificationError("complete campaign cell is missing aggregate-summary.json")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if (
        aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("outer_folds") != folds
        or aggregate.get("holdout_axis") != "genes"
        or aggregate.get("cell_holdout") is not False
        or aggregate.get("scientific_result_scope")
        != "one_split_seed_by_one_training_seed_cell"
        or aggregate.get("top_level_campaign_aggregation_required") is not True
        or set(aggregate.get("conditions", {})) != set(conditions)
    ):
        raise VerificationError("aggregate identity or condition inventory differs")
    target = np.asarray(gene_effect.T, dtype=np.float32)
    expected_condition_summaries: dict[str, Any] = {}
    for condition in conditions:
        full_prediction = np.full(target.shape, np.nan, dtype=np.float32)
        full_cell = np.full(target.shape, np.nan, dtype=np.float32)
        full_global = np.full(target.shape, np.nan, dtype=np.float32)
        for fold in folds:
            indices = fold_gene_sets[fold]
            full_prediction[:, indices] = prediction_columns[condition][fold]
            full_cell[:, indices] = cell_references[condition][fold][:, None]
            full_global[:, indices] = global_references[condition][fold]
        if (
            not np.isfinite(full_prediction).all()
            or not np.isfinite(full_cell).all()
            or not np.isfinite(full_global).all()
        ):
            raise VerificationError(f"{condition}: pooled predictions are incomplete")
        recomputed = independent_metrics(full_prediction, target, full_cell, full_global)
        observed_condition = aggregate["conditions"][condition]
        assert_metric_mapping(
            observed_condition["pooled_outer_predictions"],
            recomputed,
            context=f"aggregate {condition}",
        )
        if observed_condition.get("prediction_artifacts") != prediction_records[condition]:
            raise VerificationError(f"aggregate {condition}: prediction records differ")

        expected_condition_summaries[condition] = {
            "pooled_outer_predictions": recomputed,
            "fold_mean_sd": {
                metric: fold_descriptive_summary(
                    [per_fold_metrics[condition][fold][metric] for fold in folds]
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
            "prediction_artifacts": prediction_records[condition],
        }

    expected_comparisons: dict[str, Any] = {}
    for comparator in ("C-PLM-MLP", BASELINE_CONDITION):
        expected_comparisons[f"DepPLM_minus_{comparator}"] = {
            metric: fold_descriptive_summary(
                [
                    (
                        float(per_fold_metrics["DepPLM"][fold][metric])
                        - float(per_fold_metrics[comparator][fold][metric])
                        if per_fold_metrics["DepPLM"][fold][metric] is not None
                        and per_fold_metrics[comparator][fold][metric] is not None
                        else None
                    )
                    for fold in folds
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
    expected_aggregate = {
        "schema": AGGREGATE_SCHEMA,
        "holdout_axis": "genes",
        "cell_holdout": False,
        "outer_folds": folds,
        "conditions": expected_condition_summaries,
        "paired_fold_differences": expected_comparisons,
        "inference_note": (
            "Fold summaries are descriptive dispersion checks only. Folds and fit "
            "seeds are not independent biological replicates; repeat-aware paired "
            "summaries are produced only by the registered top-level campaign."
        ),
        "scientific_result_scope": "one_split_seed_by_one_training_seed_cell",
        "top_level_campaign_aggregation_required": True,
    }
    assert_nested_mapping(aggregate, expected_aggregate, context="aggregate")

    return {
        "status": "verified",
        "campaign": str(campaign),
        "split_seed": split_seed,
        "training_seed": training_seed,
        "device": run["device"],
        "holdout_axis": "genes",
        "cell_holdout": False,
        "folds_verified": folds,
        "conditions_verified": conditions,
        "condition_fold_metric_records_recomputed": metric_count,
        "unique_held_out_genes": int(expected_source_genes.size),
        "exact_once_per_condition": True,
        "release_manifest_sha256": sha256_file(campaign / "release-manifest.json"),
        "fold_manifest_sha256": sha256_file(persisted_manifest),
        "aggregate_summary_sha256": sha256_file(aggregate_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    report = verify_campaign(parse_args(argv).campaign)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
