"""Validate and aggregate the prespecified residual-anchor research campaign.

This campaign is deliberately separate from corrected-v2. A complete result is
four paired conditions x three protocols x three seeds. Every outer-test metric
is replayed from its hash-bound prediction artifact by the corrected-v2
validator before any comparison is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from aggregate_corrected_campaign import (
    CANONICAL_N_CELLS,
    CANONICAL_N_GENES,
    PROTOCOLS,
    REPORT_METRICS,
    CampaignSpec,
    RunKey,
    RunValidationError,
    ValidatedRun,
    _load_gene_effect,
    _mean_sd_range,
    validate_run_directory,
)
from corrected_protocol import atomic_write_json
from embedding_provenance import validate_embedding_bundle
from run_corrected_experiments import (
    CANONICAL_ARGUMENTS,
    CANONICAL_SEEDS,
    FULL_GRID,
    RESIDUAL_ANCHOR_CONDITIONS,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = ROOT / "results" / "residual_anchor_v1"
PRIMARY_METRIC = "residual_r2_vs_train_gene_mean"
SAFETY_METRIC = "mse"
PAIRED_CONDITIONS = {
    "PLM": ("RA-PLM-absolute", "RA-PLM-anchored"),
    "DepPLM": ("RA-DepPLM-absolute", "RA-DepPLM-anchored"),
}


class ResidualAnchorValidationError(RuntimeError):
    """Raised when a research run or complete campaign violates its contract."""


def residual_anchor_spec() -> CampaignSpec:
    """Return the fixed 36-run residual-anchor contract."""
    return CampaignSpec(
        protocols=tuple(PROTOCOLS),
        conditions=tuple(sorted(RESIDUAL_ANCHOR_CONDITIONS)),
        seeds=tuple(int(seed) for seed in CANONICAL_SEEDS),
        canonical_arguments=dict(CANONICAL_ARGUMENTS),
        tuning_grid=tuple(dict(config) for config in FULL_GRID),
        condition_configs={
            name: dict(config)
            for name, config in RESIDUAL_ANCHOR_CONDITIONS.items()
        },
    )


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResidualAnchorValidationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResidualAnchorValidationError(f"JSON root must be an object: {path}")
    return payload


def _target_mode(condition: str) -> str:
    try:
        return str(RESIDUAL_ANCHOR_CONDITIONS[condition]["target_mode"])
    except KeyError as exc:
        raise ResidualAnchorValidationError(
            f"condition has no residual-anchor target mode: {condition!r}"
        ) from exc


def _validate_research_metadata(run: ValidatedRun) -> None:
    target_mode = _target_mode(run.key.condition)
    manifest = _load_json(run.path / "run-manifest.json")
    expected_design = {
        "campaign": "residual_anchor_v1",
        "status": "unvalidated_hypothesis",
        "paired_factors": ["target_mode", "validation_residual_scale"],
        "target_mode": target_mode,
        "claim_policy": (
            "Do not claim improvement until the complete paired matrix is "
            "independently aggregated from outer-test predictions."
        ),
    }
    if manifest.get("research_design") != expected_design:
        raise ResidualAnchorValidationError(
            f"{run.key.label()} has invalid run-manifest research_design"
        )

    summary = _load_json(run.path / "summary.json")
    if summary.get("campaign") != "residual_anchor_v1":
        raise ResidualAnchorValidationError(
            f"{run.key.label()} is missing the residual_anchor_v1 campaign label"
        )
    if summary.get("target_mode") != target_mode:
        raise ResidualAnchorValidationError(
            f"{run.key.label()} has inconsistent summary target_mode"
        )


def validate_research_run(
    run_dir: Path,
    *,
    workspace_root: Path,
    spec: CampaignSpec,
    gene_effect: Any,
    embedding_provenance: Mapping[str, Any],
    reference_hash_cache: dict[tuple[int, ...], str],
) -> ValidatedRun:
    """Replay one completion and enforce residual-anchor-specific metadata."""
    try:
        run = validate_run_directory(
            run_dir,
            workspace_root=workspace_root,
            spec=spec,
            _gene_effect=gene_effect,
            _reference_hash_cache=reference_hash_cache,
            _expected_embedding_provenance=embedding_provenance,
        )
    except RunValidationError as exc:
        raise ResidualAnchorValidationError(
            f"[{exc.code}] {run_dir}: {exc}"
        ) from exc
    _validate_research_metadata(run)
    return run


def completion_directories(results_root: Path) -> tuple[Path, ...]:
    if not results_root.exists():
        return ()
    if not results_root.is_dir():
        raise ResidualAnchorValidationError(
            f"results root is not a directory: {results_root}"
        )
    return tuple(
        path
        for path in sorted(results_root.iterdir(), key=lambda item: item.name)
        if path.is_dir() and (path / "completion.json").is_file()
    )


def _index_runs(runs: Iterable[ValidatedRun]) -> dict[RunKey, ValidatedRun]:
    indexed: dict[RunKey, ValidatedRun] = {}
    for run in runs:
        if run.key in indexed:
            raise ResidualAnchorValidationError(
                f"duplicate completed campaign cell {run.key.label()}: "
                f"{indexed[run.key].path} and {run.path}"
            )
        indexed[run.key] = run
    return indexed


def _require_campaign_consistency(runs: Mapping[RunKey, ValidatedRun]) -> None:
    values = tuple(runs.values())
    for attribute in (
        "cohort_sha256",
        "code_hashes",
        "runtime_fingerprint",
        "runtime_fingerprint_sha256",
        "embedding_provenance_sha256",
    ):
        observed = {json.dumps(getattr(run, attribute), sort_keys=True) for run in values}
        if len(observed) != 1:
            raise ResidualAnchorValidationError(
                f"completed runs do not share one {attribute}"
            )

    input_hashes: dict[str, set[str]] = {}
    for run in values:
        for path, sha256 in run.input_hashes.items():
            input_hashes.setdefault(path, set()).add(sha256)
    conflicts = sorted(path for path, hashes in input_hashes.items() if len(hashes) != 1)
    if conflicts:
        raise ResidualAnchorValidationError(
            "input hashes conflict across completed runs: " + ", ".join(conflicts)
        )

    for protocol in PROTOCOLS:
        for seed in CANONICAL_SEEDS:
            for architecture, (absolute_name, anchored_name) in PAIRED_CONDITIONS.items():
                absolute = runs[RunKey(protocol, absolute_name, int(seed))]
                anchored = runs[RunKey(protocol, anchored_name, int(seed))]
                if absolute.splits != anchored.splits:
                    raise ResidualAnchorValidationError(
                        f"{architecture} pair does not share identical splits for "
                        f"{protocol}|seed-{seed}"
                    )
                if absolute.input_hashes != anchored.input_hashes:
                    raise ResidualAnchorValidationError(
                        f"{architecture} pair does not share identical inputs for "
                        f"{protocol}|seed-{seed}"
                    )


def _aggregate(
    runs: Mapping[RunKey, ValidatedRun], spec: CampaignSpec
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    aggregate: dict[str, Any] = {}
    seed_means: dict[tuple[str, str, str, int], float] = {}
    for protocol in spec.protocols:
        protocol_payload: dict[str, Any] = {}
        for condition in spec.conditions:
            condition_payload: dict[str, Any] = {}
            for metric in REPORT_METRICS:
                by_seed: dict[str, Any] = {}
                across_seeds: list[float] = []
                for seed in spec.seeds:
                    run = runs[RunKey(protocol, condition, seed)]
                    fold_values = [
                        run.fold_metrics[fold][metric]
                        for fold in sorted(run.fold_metrics)
                    ]
                    seed_mean = statistics.fmean(fold_values)
                    seed_means[(protocol, condition, metric, seed)] = seed_mean
                    across_seeds.append(seed_mean)
                    by_seed[str(seed)] = {
                        "n_outer_folds": len(fold_values),
                        "outer_fold_values": fold_values,
                        "mean_across_outer_folds": seed_mean,
                    }
                condition_payload[metric] = {
                    "unit_of_replication": "campaign_seed",
                    "fold_handling": "outer folds averaged within each seed",
                    "by_seed": by_seed,
                    "across_seed_summary": _mean_sd_range(across_seeds),
                }
            protocol_payload[condition] = condition_payload
        aggregate[protocol] = protocol_payload

    comparisons: dict[str, Any] = {}
    all_primary_positive = True
    all_safety_nonworse = True
    for protocol in spec.protocols:
        protocol_payload = {}
        for architecture, (absolute_name, anchored_name) in PAIRED_CONDITIONS.items():
            metric_payload = {}
            for metric, favorable in (
                (PRIMARY_METRIC, "positive"),
                (SAFETY_METRIC, "nonpositive"),
            ):
                differences = [
                    seed_means[(protocol, anchored_name, metric, seed)]
                    - seed_means[(protocol, absolute_name, metric, seed)]
                    for seed in spec.seeds
                ]
                metric_payload[metric] = {
                    "contrast": f"{anchored_name} minus {absolute_name}",
                    "favorable_direction": favorable,
                    "paired_seed_differences": {
                        str(seed): value
                        for seed, value in zip(spec.seeds, differences)
                    },
                    "descriptive_summary": _mean_sd_range(differences),
                    "sign_pattern": {
                        "positive": sum(value > 0 for value in differences),
                        "negative": sum(value < 0 for value in differences),
                        "zero": sum(value == 0 for value in differences),
                    },
                    "inferential_p_value": None,
                    "inference_reason": (
                        "The three seeds are repeated nested-split-and-training reruns, "
                        "not independent biological units."
                    ),
                }
                if metric == PRIMARY_METRIC:
                    all_primary_positive &= all(value > 0 for value in differences)
                else:
                    all_safety_nonworse &= all(value <= 0 for value in differences)
            protocol_payload[architecture] = metric_payload
        comparisons[protocol] = protocol_payload

    gates = {
        "definition": (
            "Pass only when every anchored-vs-absolute pair has positive residual-R2 "
            "delta and nonpositive MSE delta for every protocol, architecture, and seed."
        ),
        "all_primary_deltas_positive": all_primary_positive,
        "all_mse_deltas_nonpositive": all_safety_nonworse,
        "strict_internal_improvement_gate_passed": (
            all_primary_positive and all_safety_nonworse
        ),
    }
    return aggregate, comparisons, gates


def validate_campaign(results_root: Path, *, workspace_root: Path) -> dict[str, Any]:
    workspace_root = workspace_root.resolve(strict=True)
    results_root = results_root.resolve()
    spec = residual_anchor_spec()
    gene_effect = _load_gene_effect(
        workspace_root,
        n_genes=CANONICAL_N_GENES,
        n_cells=CANONICAL_N_CELLS,
    )
    embedding_provenance = validate_embedding_bundle(workspace_root)
    reference_hash_cache: dict[tuple[int, ...], str] = {}
    runs = _index_runs(
        validate_research_run(
            path,
            workspace_root=workspace_root,
            spec=spec,
            gene_effect=gene_effect,
            embedding_provenance=embedding_provenance,
            reference_hash_cache=reference_hash_cache,
        )
        for path in completion_directories(results_root)
    )
    expected = set(spec.expected_keys())
    observed = set(runs)
    if observed != expected:
        missing = sorted(key.label() for key in expected - observed)
        unexpected = sorted(key.label() for key in observed - expected)
        raise ResidualAnchorValidationError(
            f"campaign matrix mismatch; missing={missing!r}, unexpected={unexpected!r}"
        )
    _require_campaign_consistency(runs)
    aggregates, comparisons, gates = _aggregate(runs, spec)
    source_path = Path(__file__).resolve(strict=True)
    return {
        "schema": "DepPLM-residual-anchor-v1-campaign",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign": "residual_anchor_v1",
        "status": "complete_and_replayed",
        "validated_run_count": len(runs),
        "expected_run_count": len(expected),
        "protocols": list(spec.protocols),
        "conditions": list(spec.conditions),
        "seeds": list(spec.seeds),
        "primary_metric": PRIMARY_METRIC,
        "safety_metric": SAFETY_METRIC,
        "integrity_checks": {
            "outer_test_metrics_recomputed_from_bound_predictions": True,
            "paired_splits_identical": True,
            "shared_code_inputs_runtime_and_embedding_provenance": True,
        },
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
        "prespecified_internal_gates": gates,
        "claim_boundary": {
            "eligible_internal_method_claim": gates[
                "strict_internal_improvement_gate_passed"
            ],
            "nar_submission_ready": False,
            "reason": (
                "This campaign tests one internal methodological hypothesis. NAR-level "
                "novelty, direct nucleic-acid insight, independent validation, and "
                "complete manuscript/repository deposition remain separate gates."
            ),
        },
        "aggregation_source": {
            "path": "code/aggregate_residual_anchor_campaign.py",
            "bytes": source_path.stat().st_size,
            "sha256": _sha256_file(source_path),
        },
        "source_runs": [
            artifact
            for key in sorted(runs)
            for artifact in runs[key].source_artifacts
        ],
    }


def write_campaign_report(report: Mapping[str, Any], output_root: Path) -> Path:
    identifier = (
        time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        + f".{time.time_ns() % 1_000_000_000:09d}Z-residual-anchor-"
        + uuid.uuid4().hex[:8]
    )
    destination = output_root / identifier
    destination.mkdir(parents=True, exist_ok=False)
    report_path = destination / "campaign.json"
    atomic_write_json(report_path, dict(report))
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_campaign(
            args.results_root,
            workspace_root=args.workspace_root,
        )
        if args.validate_only:
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        else:
            output_root = args.output_root or (args.results_root / "campaigns")
            report_path = write_campaign_report(report, output_root)
            print(report_path)
        return 0
    except (OSError, ValueError, ResidualAnchorValidationError) as exc:
        print(f"[residual-anchor] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
