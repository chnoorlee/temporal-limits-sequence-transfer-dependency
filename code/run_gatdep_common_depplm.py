"""Run DepPLM controls on the exact audited GATDep common cohort.

This is a thin, fail-closed wrapper around the frozen corrected DepPLM runner.
It changes only the cohort: cells and output genes are restricted to the exact
1088 x 2341 intersection recorded by ``gatdep_official_baseline.py``. The
five outer folds and all nested-selection settings remain prespecified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import gatdep_official_baseline as gatdep
import run_corrected_experiments as core
from corrected_protocol import atomic_write_json, set_global_seed
from embedding_provenance import (
    EMBEDDING_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    NAMES_RELATIVE_PATH,
    NORMALIZED_RELATIVE_PATH,
    validate_embedding_bundle,
)


RUN_SCHEMA = "DepPLM-GATDep-common-cohort-run-v3"
COHORT_SCHEMA = "DepPLM-GATDep-common-cohort-v1"
SUMMARY_SCHEMA = "DepPLM-GATDep-common-cohort-summary-v3"
COMPLETION_SCHEMA = "DepPLM-GATDep-common-cohort-completion-v3"
CHECKPOINT_SCHEMA = "DepPLM-outer-test-replay-checkpoint-v3"
REPLAY_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
CHECKPOINT_POLICY = (
    "per_fold_final_model_and_preprocessing_only_with_"
    "source_reconstructed_outer_test_replay"
)
REPLAY_INPUT_POLICY = (
    "not_embedded_reconstructed_from_hash_bound_project_inputs"
)
GENERATED_ARTIFACT_REDISTRIBUTION_STATUS = (
    "not_cleared_for_public_redistribution"
)
PREPROCESSING_CONTENT_POLICY = (
    "training_fitted_aggregate_statistics_and_feature_definitions_no_raw_cell_rows"
)
SUPPORTED_CONDITIONS = ("DepPLM", "C-PLM-MLP")
FIXED_ARGUMENTS: dict[str, Any] = {
    "protocol": "p1",
    "folds": 5,
    "inner_val_fraction": 0.2,
    "inner_max_epochs": 20,
    "patience": 5,
    "batch_size": 64,
    "gene_subsample": 6_000,
    "top_mut": 5_000,
    "n_expr_pc": 128,
    "pilot": False,
}
SOURCE_PATHS = (
    "code/run_gatdep_common_depplm.py",
    "code/verify_gatdep_common_depplm.py",
    "code/run_corrected_experiments.py",
    "code/corrected_protocol.py",
    "code/models.py",
    "code/03_embed_proteins.py",
    "code/embedding_provenance.py",
    "code/gatdep_official_baseline.py",
    "code/verify_gatdep_official_baseline.py",
)
PROJECT_INPUT_PATHS = (
    "data/processed/gene_effect.npy",
    EMBEDDING_RELATIVE_PATH,
    NORMALIZED_RELATIVE_PATH,
    NAMES_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    "data/processed/gene_seqs.fa",
    "data/processed/ppi_graph.npz",
    "data/processed/genes.txt",
    "data/processed/cell_lines.txt",
    "data/processed/lineages.txt",
    "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "data/depmap/OmicsSomaticMutations.csv",
)
OFFICIAL_INPUT_PATHS = (
    "datasets/genes2ID_2k.json",
    "datasets/samples2ID.json",
)
METRIC_NAMES = (
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
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_or_match(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if load_json(path) != gatdep._json_compatible(value):
            raise gatdep.AtomicWriteError(f"existing immutable record differs: {path}")
        return
    atomic_write_json(path, value)


def immutable_timestamp(path: Path, field: str) -> str:
    if path.exists():
        value = load_json(path).get(field)
        if isinstance(value, str) and value.strip():
            return value
        raise gatdep.AtomicWriteError(f"existing record lacks {field}: {path}")
    return gatdep.utc_now()


def scoped_records(root: Path, paths: Sequence[str]) -> list[dict[str, Any]]:
    return [gatdep.file_record(root / relative, relative_to=root) for relative in paths]


def _tensor_record(value: Any) -> dict[str, Any]:
    import torch

    tensor = value.detach().cpu()
    if tensor.layout != torch.strided:
        if tensor.layout != torch.sparse_coo:
            raise TypeError(f"unsupported checkpoint tensor layout: {tensor.layout}")
        tensor = tensor.coalesce()
        return {
            "layout": str(tensor.layout),
            "shape": list(tensor.shape),
            "indices": _tensor_record(tensor.indices()),
            "values": _tensor_record(tensor.values()),
        }
    array = np.ascontiguousarray(tensor.numpy())
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return {
        "layout": "torch.strided",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def _checkpoint_inventory(value: Any) -> Any:
    import torch

    if type(value) is torch.Tensor:
        return _tensor_record(value)
    if type(value) is dict:
        if not all(isinstance(key, str) for key in value):
            raise TypeError("checkpoint mapping keys must all be strings")
        return {
            key: _checkpoint_inventory(item)
            for key, item in sorted(value.items())
        }
    if type(value) in (tuple, list):
        return [_checkpoint_inventory(item) for item in value]
    if type(value) is float and not np.isfinite(value):
        raise ValueError("checkpoint scalar values must be finite")
    if type(value) in (str, int, float, bool) or value is None:
        return value
    raise TypeError(f"unsupported checkpoint inventory value: {type(value)!r}")


def checkpoint_component_hash(value: Any) -> str:
    return gatdep.canonical_hash(_checkpoint_inventory(value))


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise gatdep.AtomicWriteError(f"refusing to overwrite checkpoint: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def runtime_record() -> dict[str, Any]:
    import pandas as pd
    import scipy
    import sklearn
    import torch

    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "module_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def build_common_base(cohort: gatdep.CommonCohort, *, seed: int) -> core.BaseData:
    full = core.load_base_data(
        pilot=False,
        seed=seed,
        pilot_cells=1,
        pilot_genes=1,
    )
    if full.cell_ids != gatdep.read_lines(core.PROCESSED / "cell_lines.txt"):
        raise RuntimeError(
            "corrected runner cell order changed during common subsetting"
        )
    if full.genes != gatdep.read_lines(core.PROCESSED / "genes.txt"):
        raise RuntimeError(
            "corrected runner gene order changed during common subsetting"
        )
    cells = cohort.project_cell_indices
    genes = cohort.project_gene_indices
    base = core.BaseData(
        gene_effect=np.asarray(
            full.gene_effect[np.ix_(genes, cells)], dtype=np.float32
        ),
        expression_raw=np.asarray(full.expression_raw[cells], dtype=np.float32),
        mutations_by_cell=[full.mutations_by_cell[int(index)] for index in cells],
        embeddings=np.asarray(full.embeddings[genes], dtype=np.float32),
        edge_index=core._subgraph(full.edge_index, genes, len(full.genes)),
        genes=list(cohort.gene_ids),
        cell_ids=list(cohort.cell_ids),
        lineages=np.asarray(full.lineages[cells], dtype=str),
        source_cell_indices=cells.copy(),
        source_gene_indices=genes.copy(),
        expression_gene_names=list(full.expression_gene_names),
    )
    if base.gene_effect.shape != (2_341, 1_088):
        raise RuntimeError(f"unexpected common target shape: {base.gene_effect.shape}")
    return base


def cohort_manifest(cohort: gatdep.CommonCohort, base: core.BaseData) -> dict[str, Any]:
    return {
        "schema": COHORT_SCHEMA,
        "scientific_result": False,
        "n_cells": len(base.cell_ids),
        "n_genes": len(base.genes),
        "n_expression_features_raw": int(base.expression_raw.shape[1]),
        "n_string_edges": int(base.edge_index.shape[0]),
        "cohort_contract": cohort.contract(),
        "cohort_contract_sha256": cohort.contract()["sha256"],
        "genes_sha256": core._hash_strings(base.genes),
        "cell_ids_sha256": core._hash_strings(base.cell_ids),
        "expression_gene_names_sha256": core._hash_strings(base.expression_gene_names),
        "source_cell_indices": base.source_cell_indices,
        "source_gene_indices": base.source_gene_indices,
    }


def fixed_core_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        **FIXED_ARGUMENTS,
        condition=args.condition,
        seed=int(args.seed),
        fold_id=int(args.fold_id),
        device=args.device,
        pilot_cells=1,
        pilot_genes=1,
    )


def _checkpoint_payload(
    *,
    artifact: core.FinalArtifact,
    result: Mapping[str, Any],
    base: core.BaseData,
    condition: str,
    fold_name: str,
    fold_number: int,
    seed: int,
    batch_size: int,
    source_input_inventory_sha256: str,
) -> dict[str, Any]:
    import torch

    fold_data = artifact.fold_data
    model_state = {
        str(name): tensor.detach().cpu().clone()
        for name, tensor in artifact.model.state_dict().items()
    }
    expression_state = {
        "input_feature_names": tuple(base.expression_gene_names),
        "mean": torch.from_numpy(np.asarray(fold_data.expression_fit.mean_).copy()),
        "scale": torch.from_numpy(np.asarray(fold_data.expression_fit.scale_).copy()),
        "pca_mean": torch.from_numpy(
            np.asarray(fold_data.expression_fit.pca_mean_).copy()
        ),
        "components": torch.from_numpy(
            np.asarray(fold_data.expression_fit.components_).copy()
        ),
        "explained_variance_ratio": torch.from_numpy(
            np.asarray(fold_data.expression_fit.explained_variance_ratio_).copy()
        ),
        "n_output_components": int(fold_data.expression_fit.n_output_components),
    }
    preprocessing_state = {
        "fit_indices": torch.from_numpy(fold_data.fit_indices.astype(np.int64, copy=True)),
        "expression": expression_state,
        "mutation": {
            "vocabulary": tuple(fold_data.mutation.vocabulary),
            "token_order": tuple(fold_data.mutation.token_order),
            "k_tokens": int(fold_data.mutation.k_tokens),
        },
        "graph": {
            "edge_index": fold_data.edge_index.detach().cpu().clone(),
            "edge_weight": fold_data.edge_weight.detach().cpu().clone(),
        },
        "lineage": {"categories": tuple(fold_data.lineage_fit.categories)},
        "selectivity_labels": torch.from_numpy(
            np.asarray(fold_data.selectivity_labels).copy()
        ),
    }
    final_config = dict(artifact.config)
    model_spec = {
        "model": str(final_config["model"]),
        "n_genes": int(len(base.genes)),
        "embedding_dim": int(model_state["emb"].shape[1]),
        "n_mutation_features": int(fold_data.mutation.matrix.shape[1]),
        "n_expression_features": int(fold_data.expression.shape[1]),
        "n_lineage_features": int(fold_data.lineage.shape[1]),
        "hidden": int(final_config["hidden"]),
        "k_tokens": int(core.K_TOKENS),
        "use_pretrained": bool(final_config.get("use_pretrained", True)),
        "use_gnn": bool(final_config.get("use_gnn", True)),
        "use_comut": bool(final_config.get("use_comut", True)),
        "use_multitask": bool(final_config.get("use_multitask", True)),
        "use_crossattn": bool(final_config.get("use_crossattn", True)),
        "freeze_embeddings": True,
    }
    prediction_record = result.get("outer_test_predictions", {})
    if not isinstance(prediction_record, Mapping):
        raise RuntimeError("fold result lacks an outer-test prediction record")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "fold": fold_name,
        "fold_number": int(fold_number),
        "condition": condition,
        "seed": int(seed),
        "fold_seed": int(result["fold_seed"]),
        "final_config": final_config,
        "selected_epoch": int(result["best_epoch"]),
        "trained_epochs": int(artifact.training["trained_epochs"]),
        "batch_size": int(batch_size),
        "prediction_sha256": str(prediction_record["sha256"]),
        "elementwise_tolerance": dict(REPLAY_TOLERANCE),
        "model_spec": model_spec,
        "model_state_dict": model_state,
        "preprocessing_state": preprocessing_state,
        "contains_cell_level_replay_inputs": False,
        "preprocessing_contains_raw_cell_level_values": False,
        "preprocessing_contains_source_derived_statistics": True,
        "preprocessing_content_policy": PREPROCESSING_CONTENT_POLICY,
        "replay_input_policy": REPLAY_INPUT_POLICY,
        "source_input_inventory_sha256": source_input_inventory_sha256,
        "redistribution_status": GENERATED_ARTIFACT_REDISTRIBUTION_STATUS,
        "redistribution_risk": (
            "model_weights_and_preprocessing_are_derived_from_DepMap_inputs"
        ),
        "model_state_sha256": checkpoint_component_hash(model_state),
        "preprocessing_state_sha256": checkpoint_component_hash(
            preprocessing_state
        ),
    }


def persist_fold_checkpoint(
    *,
    run_dir: Path,
    artifact: core.FinalArtifact,
    result: Mapping[str, Any],
    base: core.BaseData,
    condition: str,
    fold_name: str,
    fold_number: int,
    seed: int,
    batch_size: int,
    source_input_inventory_sha256: str,
) -> dict[str, Any]:
    payload = _checkpoint_payload(
        artifact=artifact,
        result=result,
        base=base,
        condition=condition,
        fold_name=fold_name,
        fold_number=fold_number,
        seed=seed,
        batch_size=batch_size,
        source_input_inventory_sha256=source_input_inventory_sha256,
    )
    _checkpoint_inventory(payload)
    path = run_dir / f"{fold_name}-checkpoint.pt"
    _atomic_torch_save(path, payload)
    return {
        "schema": CHECKPOINT_SCHEMA,
        **gatdep.file_record(path, relative_to=run_dir),
        "format": "torch_save_weights_only_compatible",
        "weights_only_load_required": True,
        "final_config": payload["final_config"],
        "selected_epoch": payload["selected_epoch"],
        "trained_epochs": payload["trained_epochs"],
        "prediction_sha256": payload["prediction_sha256"],
        "model_state_sha256": payload["model_state_sha256"],
        "preprocessing_state_sha256": payload["preprocessing_state_sha256"],
        "contains_cell_level_replay_inputs": False,
        "preprocessing_contains_raw_cell_level_values": False,
        "preprocessing_contains_source_derived_statistics": True,
        "preprocessing_content_policy": payload[
            "preprocessing_content_policy"
        ],
        "replay_input_policy": payload["replay_input_policy"],
        "source_input_inventory_sha256": payload[
            "source_input_inventory_sha256"
        ],
        "redistribution_status": payload["redistribution_status"],
        "elementwise_tolerance": payload["elementwise_tolerance"],
    }


def run_fold_with_checkpoint(
    *,
    fold_name: str,
    fold_number: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    base: core.BaseData,
    condition: str,
    core_args: argparse.Namespace,
    device: str,
    run_dir: Path,
    source_input_inventory_sha256: str,
) -> dict[str, Any]:
    artifact_paths = (
        run_dir / f"{fold_name}-split.json",
        run_dir / f"{fold_name}-result.json",
        run_dir / f"{fold_name}-predictions.npz",
        run_dir / f"{fold_name}-checkpoint.pt",
    )
    existing = [path.name for path in artifact_paths if path.exists()]
    if existing:
        raise gatdep.AtomicWriteError(
            "refusing to overwrite existing fold artifacts: " + ", ".join(existing)
        )

    captured: list[core.FinalArtifact] = []
    persisted_results: list[dict[str, Any]] = []
    original_nested_selection = core.run_nested_selection
    original_atomic_write_json = core.atomic_write_json

    def capture_final_artifact(*args: Any, **kwargs: Any) -> Any:
        selected = original_nested_selection(*args, **kwargs)
        artifact = selected.final_model
        if not isinstance(artifact, core.FinalArtifact):
            raise TypeError("nested selection did not return a FinalArtifact")
        captured.append(artifact)
        return selected

    def persist_result_with_checkpoint(path: Path, value: Any) -> None:
        result_path = run_dir / f"{fold_name}-result.json"
        if Path(path) != result_path:
            original_atomic_write_json(path, value)
            return
        if len(captured) != 1:
            raise RuntimeError(
                f"expected one captured final model for {fold_name}, got {len(captured)}"
            )
        if not isinstance(value, Mapping):
            raise TypeError("core fold result must be a mapping")
        result = dict(value)
        checkpoint = persist_fold_checkpoint(
            run_dir=run_dir,
            artifact=captured[0],
            result=result,
            base=base,
            condition=condition,
            fold_name=fold_name,
            fold_number=fold_number,
            seed=int(core_args.seed),
            batch_size=int(core_args.batch_size),
            source_input_inventory_sha256=source_input_inventory_sha256,
        )
        result.update(
            {
                "outer_test_replay_checkpoint": checkpoint,
                "checkpoint_replay_required": True,
            }
        )
        original_atomic_write_json(path, result)
        persisted_results.append(result)

    core.run_nested_selection = capture_final_artifact
    core.atomic_write_json = persist_result_with_checkpoint
    try:
        core.run_fold(
            fold_name=fold_name,
            fold_number=fold_number,
            outer_train=outer_train,
            outer_test=outer_test,
            base=base,
            condition=condition,
            base_config=core.CONDITIONS[condition],
            grid=core.FULL_GRID,
            args=core_args,
            device=device,
            run_dir=run_dir,
        )
    finally:
        core.run_nested_selection = original_nested_selection
        core.atomic_write_json = original_atomic_write_json
    if len(persisted_results) != 1:
        raise RuntimeError(
            f"expected one persisted result for {fold_name}, got {len(persisted_results)}"
        )
    return persisted_results[0]


def summarize_if_complete(
    run_dir: Path,
    *,
    condition: str,
    seed: int,
    cohort_hash: str,
    split_hash: str,
) -> None:
    result_paths = [run_dir / f"fold-{fold:02d}-result.json" for fold in range(5)]
    if not all(path.is_file() for path in result_paths):
        return
    results = [load_json(path) for path in result_paths]
    coverage: list[int] = []
    artifacts: list[dict[str, Any]] = []
    for fold in range(5):
        for suffix, extension in (
            ("split", "json"),
            ("result", "json"),
            ("predictions", "npz"),
            ("checkpoint", "pt"),
        ):
            artifacts.append(
                gatdep.file_record(
                    run_dir / f"fold-{fold:02d}-{suffix}.{extension}",
                    relative_to=run_dir,
                )
            )
        with np.load(
            run_dir / f"fold-{fold:02d}-predictions.npz", allow_pickle=False
        ) as bundle:
            coverage.extend(bundle["outer_test_indices"].tolist())
    if sorted(coverage) != list(range(1_088)):
        raise RuntimeError("outer-test prediction files do not cover each cell once")

    metrics: dict[str, dict[str, float | int | None]] = {}
    for name in METRIC_NAMES:
        values = np.asarray(
            [result.get("test_metrics", {}).get(name, np.nan) for result in results],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        metrics[name] = {
            "mean": float(np.mean(finite)) if finite.size else None,
            "sd": float(np.std(finite)) if finite.size else None,
            "n": int(finite.size),
        }
    summary_path = run_dir / "summary.json"
    summary = {
        "schema": SUMMARY_SCHEMA,
        "completed_at_utc": immutable_timestamp(summary_path, "completed_at_utc"),
        "scientific_result": False,
        "condition": condition,
        "seed": int(seed),
        "completed_folds": list(range(5)),
        "outer_test_cells_exactly_once": True,
        "checkpoint_replay_required": True,
        "cohort_contract_sha256": cohort_hash,
        "outer_split_contract_sha256": split_hash,
        "primary_metric": "residual_r2_vs_train_gene_mean",
        "metrics": metrics,
        "artifacts": artifacts,
    }
    write_or_match(summary_path, summary)
    completion_path = run_dir / "completion.json"
    completion = {
        "schema": COMPLETION_SCHEMA,
        "completed_at_utc": immutable_timestamp(completion_path, "completed_at_utc"),
        "scientific_result": False,
        "protocol_complete": True,
        "independent_verification_required": True,
        "checkpoint_replay_required": True,
        "run_manifest_sha256": gatdep.sha256_file(run_dir / "run-manifest.json"),
        "cohort_manifest_sha256": gatdep.sha256_file(run_dir / "cohort-manifest.json"),
        "summary_sha256": gatdep.sha256_file(summary_path),
        "artifacts": artifacts,
    }
    write_or_match(completion_path, completion)


def run(args: argparse.Namespace) -> Path:
    project_root = Path(args.project_root).resolve()
    official_root = Path(args.official_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    if project_root != core.ROOT.resolve():
        raise ValueError(
            f"--project-root must match the imported frozen core root {core.ROOT}"
        )
    if gatdep._git(official_root, "rev-parse", "HEAD") != gatdep.OFFICIAL_COMMIT:
        raise ValueError("official checkout is not at the pinned GATDep commit")
    if gatdep._git(official_root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("official checkout has tracked modifications")
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()) and not (run_dir / "run-manifest.json").exists():
        raise gatdep.AtomicWriteError("refusing a non-empty unbound run directory")

    validate_embedding_bundle(project_root)
    cohort = gatdep.build_common_cohort(project_root, official_root)
    if (len(cohort.cell_ids), len(cohort.gene_ids)) != (1_088, 2_341):
        raise RuntimeError("common cohort is not the audited 1088 x 2341 endpoint")
    split_contract = gatdep.outer_split_contract(cohort, seed=args.seed, n_splits=5)
    source_records = scoped_records(project_root, SOURCE_PATHS)
    project_inputs = scoped_records(project_root, PROJECT_INPUT_PATHS)
    official_inputs = scoped_records(official_root, OFFICIAL_INPUT_PATHS)
    device = core._device_from_arg(args.device)
    manifest_path = run_dir / "run-manifest.json"
    manifest = {
        "schema": RUN_SCHEMA,
        "created_at_utc": immutable_timestamp(manifest_path, "created_at_utc"),
        "scientific_result": False,
        "condition": args.condition,
        "seed": int(args.seed),
        "planned_folds": list(range(5)),
        "fixed_arguments": FIXED_ARGUMENTS,
        "tuning_grid": core.FULL_GRID,
        "condition_configuration": core.CONDITIONS[args.condition],
        "endpoint": "raw_DepMap_gene_effect_common_2341_genes",
        "outer_test_policy": "exactly_once_per_fold_after_fixed_epoch_refit",
        "checkpoint_policy": CHECKPOINT_POLICY,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_replay_tolerance": REPLAY_TOLERANCE,
        "checkpoint_contains_cell_level_replay_inputs": False,
        "checkpoint_preprocessing_contains_raw_cell_level_values": False,
        "checkpoint_preprocessing_contains_source_derived_statistics": True,
        "checkpoint_preprocessing_content_policy": PREPROCESSING_CONTENT_POLICY,
        "replay_input_policy": REPLAY_INPUT_POLICY,
        "generated_artifact_redistribution_status": (
            GENERATED_ARTIFACT_REDISTRIBUTION_STATUS
        ),
        "execution_device": device,
        "cohort_contract": cohort.contract(),
        "cohort_contract_sha256": cohort.contract()["sha256"],
        "outer_split_contract": split_contract,
        "outer_split_contract_sha256": split_contract["sha256"],
        "official_repository": gatdep.OFFICIAL_REPOSITORY,
        "official_commit": gatdep.OFFICIAL_COMMIT,
        "official_code_executed": False,
        "third_party_redistribution_status": gatdep.REDISTRIBUTION_LICENSE_STATUS,
        "third_party_materials_in_public_release": False,
        "runtime": runtime_record(),
        "source_files": source_records,
        "project_input_files": project_inputs,
        "official_input_files": official_inputs,
    }
    write_or_match(manifest_path, manifest)

    base = build_common_base(cohort, seed=args.seed)
    write_or_match(run_dir / "cohort-manifest.json", cohort_manifest(cohort, base))
    set_global_seed(args.seed)
    core_args = fixed_core_args(args)
    folds = core.make_kfold_outer_splits(1_088, 5, args.seed)
    selected = range(5) if args.fold_id == -1 else [args.fold_id]
    for fold_number in selected:
        outer_train, outer_test = folds[fold_number]
        run_fold_with_checkpoint(
            fold_name=f"fold-{fold_number:02d}",
            fold_number=fold_number,
            outer_train=outer_train,
            outer_test=outer_test,
            base=base,
            condition=args.condition,
            core_args=core_args,
            device=device,
            run_dir=run_dir,
            source_input_inventory_sha256=gatdep.canonical_hash(project_inputs),
        )

    current_sources = scoped_records(project_root, SOURCE_PATHS)
    current_project_inputs = scoped_records(project_root, PROJECT_INPUT_PATHS)
    current_official_inputs = scoped_records(official_root, OFFICIAL_INPUT_PATHS)
    if (
        current_sources != source_records
        or current_project_inputs != project_inputs
        or current_official_inputs != official_inputs
    ):
        raise RuntimeError("source or input files changed during the formal run")
    summarize_if_complete(
        run_dir,
        condition=args.condition,
        seed=args.seed,
        cohort_hash=cohort.contract()["sha256"],
        split_hash=split_contract["sha256"],
    )
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parent.parent)
    )
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--condition", choices=SUPPORTED_CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-id", type=int, choices=(-1, 0, 1, 2, 3, 4), default=-1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = run(args)
    print(json.dumps({"run_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
