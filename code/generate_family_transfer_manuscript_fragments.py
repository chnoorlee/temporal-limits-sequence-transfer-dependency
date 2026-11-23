"""Generate hash-bound manuscript fragments from verified family-transfer results.

This utility is deliberately read-only with respect to scientific results.  It
accepts only the terminal aggregate and temporal result schemas, verifies the
metrics files against their completion records, and writes a compact provenance
record plus LaTeX macros and a comparison table.  It never recomputes, filters,
or selects scientific results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


CAMPAIGN_COMPLETION_SCHEMA = "DepPLM-family-inductive-campaign-completion-v2"
CAMPAIGN_METRICS_SCHEMA = "DepPLM-family-inductive-campaign-v2"
TEMPORAL_COMPLETION_SCHEMA = "DepPLM-temporal-26q3-completion-v1"
TEMPORAL_METRICS_SCHEMA = "DepPLM-temporal-26q3-result-v1"
OUTPUT_SCHEMA = "DepPLM-family-transfer-manuscript-fragments-v1"


class FragmentGenerationError(ValueError):
    """Raised when an input is not the bound terminal result expected here."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FragmentGenerationError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FragmentGenerationError(f"expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FragmentGenerationError(message)


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FragmentGenerationError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FragmentGenerationError(f"{label} must be finite")
    return numeric


def _require_finite_tree(value: Any, label: str = "result") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _require_number(value, label)


def _verify_bound_file(
    directory: Path,
    completion: Mapping[str, Any],
    artifact_key: str,
    expected_filename: str,
) -> tuple[Path, str]:
    artifacts = completion.get("artifacts")
    _require(isinstance(artifacts, dict), "completion artifacts must be an object")
    record = artifacts.get(artifact_key)
    _require(isinstance(record, dict), f"missing artifact record: {artifact_key}")
    _require(record.get("path") == expected_filename, f"unexpected path for {artifact_key}")
    expected_hash = record.get("sha256")
    expected_bytes = record.get("bytes")
    _require(isinstance(expected_hash, str) and len(expected_hash) == 64, "invalid SHA-256 record")
    _require(isinstance(expected_bytes, int) and expected_bytes >= 0, "invalid byte-count record")
    path = directory / expected_filename
    _require(path.is_file() and not path.is_symlink(), f"missing regular artifact: {path}")
    _require(path.stat().st_size == expected_bytes, f"byte-count mismatch: {path}")
    actual_hash = _sha256(path)
    _require(actual_hash == expected_hash, f"SHA-256 mismatch: {path}")
    return path, actual_hash


def _campaign_bundle(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    completion_path = directory / "completion.json"
    completion = _load_json(completion_path)
    _require(completion.get("schema") == CAMPAIGN_COMPLETION_SCHEMA, "unexpected campaign completion schema")
    _require(completion.get("status") == "complete", "campaign is not complete")
    _require(completion.get("external_outcomes_read") is False, "campaign reports external outcome access")
    integrity = completion.get("integrity")
    _require(isinstance(integrity, dict), "campaign integrity record is absent")
    for key in (
        "complete_45_job_cartesian_product",
        "completion_written_last",
        "each_gene_exactly_once_per_seed_pair",
        "each_job_baseline_selected_from_its_own_inner_validation_only",
        "nine_predictions_per_gene",
    ):
        _require(integrity.get(key) is True, f"campaign integrity flag is not true: {key}")
    metrics_path, metrics_hash = _verify_bound_file(
        directory, completion, "campaign_metrics", "campaign-metrics.json"
    )
    metrics = _load_json(metrics_path)
    _require(metrics.get("schema") == CAMPAIGN_METRICS_SCHEMA, "unexpected campaign metrics schema")
    _require(metrics.get("status") == "complete_and_replayed", "campaign metrics were not independently replayed")
    return completion, metrics, {
        "completion_sha256": _sha256(completion_path),
        "metrics_sha256": metrics_hash,
    }


def _temporal_bundle(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    completion_path = directory / "completion.json"
    completion = _load_json(completion_path)
    _require(completion.get("schema") == TEMPORAL_COMPLETION_SCHEMA, "unexpected temporal completion schema")
    _require(
        completion.get("status") in {"complete", "blocked_unresolved_organoid_overlap"},
        "temporal analysis is not terminal",
    )
    _require(completion.get("all_failed_gates_retained") is True, "failed temporal gates were not retained")
    _require(
        completion.get("primary_post_access_refit_or_recalibration") is False,
        "primary temporal result reports post-access refitting or recalibration",
    )
    metrics_path, metrics_hash = _verify_bound_file(directory, completion, "metrics", "metrics.json")
    metrics = _load_json(metrics_path)
    _require(metrics.get("schema") == TEMPORAL_METRICS_SCHEMA, "unexpected temporal metrics schema")
    _require(
        metrics.get("primary_post_access_refit_or_recalibration") is False,
        "temporal metrics report post-access refitting or recalibration",
    )
    return completion, metrics, {
        "completion_sha256": _sha256(completion_path),
        "metrics_sha256": metrics_hash,
    }


def extract_results(campaign_dir: Path, temporal_dir: Path) -> dict[str, Any]:
    campaign_completion, campaign, campaign_hashes = _campaign_bundle(campaign_dir)
    temporal_completion, temporal, temporal_hashes = _temporal_bundle(temporal_dir)

    primary = campaign["primary_analysis"]
    proposed = primary["methods"]["FAMILY-DISTANCE-STACK"]
    baseline = primary["per_job_selected_baseline"]
    bootstrap = campaign["component_block_bootstrap"]
    delta = bootstrap["delta_r2_vs_consensus_per_job_selected_baseline"]
    global_gain = bootstrap["mse_improvement_vs_consensus_global"]
    context = campaign["secondary_context"]
    context_bootstrap = context["component_block_bootstrap_r2"]
    temporal_primary = temporal["primary_temporal_cohort"]
    temporal_proposed = temporal_primary["primary_method"]
    temporal_baseline = temporal_primary["selected_baseline"]
    temporal_bootstrap = temporal_primary["component_block_bootstrap"][
        "delta_r2_vs_frozen_selected_baseline"
    ]
    correction = temporal["shared_model_pipeline_correction_sensitivity"]
    correction_bootstrap = correction["component_block_bootstrap"][
        "delta_r2_vs_corrected_selected_baseline"
    ]
    overlap = temporal["organoid_overlap_audit"]

    for label, value in {
        "development proposed R2": proposed["r2_vs_consensus_global_train_mean"],
        "development baseline R2": baseline["r2_vs_consensus_global_train_mean"],
        "development delta R2": delta["estimate"],
        "temporal proposed R2": temporal_proposed["r2_vs_frozen_consensus_global"],
        "temporal baseline R2": temporal_baseline["r2_vs_frozen_consensus_global"],
        "temporal delta R2": temporal_primary["delta_r2_vs_frozen_selected_baseline"],
    }.items():
        _require_number(value, label)

    return {
        "schema": OUTPUT_SCHEMA,
        "source": {
            "campaign_id": campaign_completion["campaign_id"],
            "campaign_completion_sha256": campaign_hashes["completion_sha256"],
            "campaign_metrics_sha256": campaign_hashes["metrics_sha256"],
            "temporal_analysis_id": temporal_completion["analysis_id"],
            "temporal_completion_sha256": temporal_hashes["completion_sha256"],
            "temporal_metrics_sha256": temporal_hashes["metrics_sha256"],
            "generator_sha256": _sha256(Path(__file__).resolve()),
        },
        "development": {
            "n_genes": proposed["n_genes"],
            "n_components": bootstrap["n_components"],
            "prediction_count_per_gene": primary["prediction_count_per_gene"],
            "proposed": proposed,
            "selected_baseline": baseline,
            "delta_r2": delta,
            "mse_improvement_vs_global": global_gain,
            "baseline_identity_counts": primary["job_baseline_selection"][
                "identity_distribution_descriptive_only"
            ],
            "gates": campaign["locked_gates"],
        },
        "context": {
            "r2_vs_zero": context["r2_vs_zero"],
            "bootstrap_r2": context_bootstrap,
            "mean_within_gene_pearson": context["mean_within_gene_pearson"],
            "mean_within_gene_spearman": context["mean_within_gene_spearman"],
            "n_finite_entries": context["n_finite_entries"],
        },
        "temporal_primary": {
            "cohort": temporal_primary["cohort"],
            "proposed": temporal_proposed,
            "selected_baseline": temporal_baseline,
            "delta_r2": temporal_primary["delta_r2_vs_frozen_selected_baseline"],
            "delta_r2_bootstrap": temporal_bootstrap,
            "gates": temporal_primary["gates"],
        },
        "shared_model_correction_sensitivity": {
            "status": correction["status"],
            "n_shared_models": correction["n_shared_models"],
            "n_eligible_genes": correction["n_eligible_genes"],
            "proposed": correction["primary_method"],
            "selected_baseline": correction["selected_baseline"],
            "delta_r2": correction["delta_r2_vs_corrected_selected_baseline"],
            "delta_r2_bootstrap": correction_bootstrap,
            "uses_26q3_outcomes": temporal[
                "secondary_shared_model_correction_uses_26q3_outcomes"
            ],
        },
        "organoid_overlap": {
            "status": overlap["audit_status"],
            "n_temporal_models": overlap["n_temporal_models"],
            "n_organoid_source_models": overlap["n_organoid_source_models"],
            "n_overlap_models": overlap["n_overlap_models"],
            "raw_exact_string_intersection_count": overlap[
                "raw_exact_string_intersection_count"
            ],
            "independence_claim_forbidden": overlap["independence_claim_forbidden"],
        },
        "claim_boundary": {
            "method_superiority_supported": campaign["locked_gates"]["method"],
            "context_prediction_supported": campaign["locked_gates"]["context"],
            "temporal_confirmation_supported": temporal_primary["gates"]["all_passed"],
            "organoid_independence_supported": not overlap["independence_claim_forbidden"],
            "nar_submission_ready": False,
        },
    }


def _f(value: Any, digits: int = 6) -> str:
    return f"{_require_number(value, 'manuscript value'):.{digits}f}"


def render_macros(result: Mapping[str, Any]) -> str:
    dev = result["development"]
    ctx = result["context"]
    temp = result["temporal_primary"]
    corr = result["shared_model_correction_sensitivity"]
    overlap = result["organoid_overlap"]
    macros = {
        "DevGenes": str(dev["n_genes"]),
        "DevComponents": str(dev["n_components"]),
        "DevPredictionsPerGene": str(dev["prediction_count_per_gene"]),
        "DevMethodRtwo": _f(dev["proposed"]["r2_vs_consensus_global_train_mean"]),
        "DevMethodMSE": _f(dev["proposed"]["mse"]),
        "DevMethodPearson": _f(dev["proposed"]["pearson"]),
        "DevBaselineRtwo": _f(dev["selected_baseline"]["r2_vs_consensus_global_train_mean"]),
        "DevBaselineMSE": _f(dev["selected_baseline"]["mse"]),
        "DevBaselinePearson": _f(dev["selected_baseline"]["pearson"]),
        "DevDeltaRtwo": _f(dev["delta_r2"]["estimate"]),
        "DevDeltaRtwoLow": _f(dev["delta_r2"]["ci_low"]),
        "DevDeltaRtwoHigh": _f(dev["delta_r2"]["ci_high"]),
        "DevGlobalMSEGain": _f(dev["mse_improvement_vs_global"]["estimate"]),
        "DevGlobalMSEGainLow": _f(dev["mse_improvement_vs_global"]["ci_low"]),
        "DevGlobalMSEGainHigh": _f(dev["mse_improvement_vs_global"]["ci_high"]),
        "ContextRtwo": _f(ctx["r2_vs_zero"]),
        "ContextRtwoLow": _f(ctx["bootstrap_r2"]["ci_low"]),
        "ContextRtwoHigh": _f(ctx["bootstrap_r2"]["ci_high"]),
        "ContextPearson": _f(ctx["mean_within_gene_pearson"]),
        "TemporalModels": str(temp["cohort"]["n_new_models"]),
        "TemporalGenes": str(temp["cohort"]["n_eligible_genes"]),
        "TemporalComponents": str(temp["cohort"]["n_sequence_components"]),
        "TemporalMethodRtwo": _f(temp["proposed"]["r2_vs_frozen_consensus_global"]),
        "TemporalMethodMSE": _f(temp["proposed"]["mse"]),
        "TemporalMethodPearson": _f(temp["proposed"]["pearson"]),
        "TemporalBaselineRtwo": _f(temp["selected_baseline"]["r2_vs_frozen_consensus_global"]),
        "TemporalBaselineMSE": _f(temp["selected_baseline"]["mse"]),
        "TemporalBaselinePearson": _f(temp["selected_baseline"]["pearson"]),
        "TemporalDeltaRtwo": _f(temp["delta_r2"]),
        "TemporalDeltaRtwoLow": _f(temp["delta_r2_bootstrap"]["ci_low"]),
        "TemporalDeltaRtwoHigh": _f(temp["delta_r2_bootstrap"]["ci_high"]),
        "SharedModels": str(corr["n_shared_models"]),
        "CorrectedMethodRtwo": _f(corr["proposed"]["r2_vs_corrected_consensus_global"]),
        "CorrectedBaselineRtwo": _f(corr["selected_baseline"]["r2_vs_corrected_consensus_global"]),
        "CorrectedDeltaRtwo": _f(corr["delta_r2"]),
        "CorrectedDeltaRtwoLow": _f(corr["delta_r2_bootstrap"]["ci_low"]),
        "CorrectedDeltaRtwoHigh": _f(corr["delta_r2_bootstrap"]["ci_high"]),
        "OrganoidSourceModels": str(overlap["n_organoid_source_models"]),
    }
    lines = [
        "% Generated by code/generate_family_transfer_manuscript_fragments.py.",
        "% Do not edit: regenerate from the hash-bound server result artifacts.",
    ]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items())
    return "\n".join(lines) + "\n"


def render_table(result: Mapping[str, Any]) -> str:
    dev = result["development"]
    temp = result["temporal_primary"]
    corr = result["shared_model_correction_sensitivity"]
    rows = (
        (
            "24Q4 development",
            dev["proposed"],
            dev["selected_baseline"],
            "r2_vs_consensus_global_train_mean",
            dev["delta_r2"]["estimate"],
        ),
        (
            "26Q3 new-model time-out",
            temp["proposed"],
            temp["selected_baseline"],
            "r2_vs_frozen_consensus_global",
            temp["delta_r2"],
        ),
        (
            "26Q3 shared-model correction sensitivity",
            corr["proposed"],
            corr["selected_baseline"],
            "r2_vs_corrected_consensus_global",
            corr["delta_r2"],
        ),
    )
    lines = [
        "% Generated from hash-bound result artifacts; do not edit.",
        "\\begin{table*}[!ht]",
        "\\caption{Full-coverage pan-context gene-mean performance. Positive $\\Delta R^2$ favors FAMILY-DISTANCE-STACK.}",
        "\\label{tab:primary-comparison}",
        "\\centering",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Analysis & Method $R^2$ & Baseline $R^2$ & $\\Delta R^2$ & Method MSE & Baseline MSE & Method $r$ & Baseline $r$ \\\\",
        "\\midrule",
    ]
    for label, proposed, baseline, r2_key, delta in rows:
        lines.append(
            f"{label} & {_f(proposed[r2_key])} & {_f(baseline[r2_key])} & "
            f"{_f(delta)} & {_f(proposed['mse'])} & {_f(baseline['mse'])} & "
            f"{_f(proposed['pearson'])} & {_f(baseline['pearson'])} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table*}"))
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate(campaign_dir: Path, temporal_dir: Path, output_dir: Path) -> dict[str, Any]:
    result = extract_results(campaign_dir.resolve(), temporal_dir.resolve())
    _require_finite_tree(result)
    json_bytes = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _atomic_write(output_dir / "family_transfer_results.json", json_bytes)
    _atomic_write(output_dir / "generated_results_macros.tex", render_macros(result).encode("ascii"))
    _atomic_write(output_dir / "generated_primary_results_table.tex", render_table(result).encode("ascii"))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--temporal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = generate(args.campaign_dir, args.temporal_dir, args.output_dir)
    except (OSError, KeyError, FragmentGenerationError) as exc:
        print(f"[family-transfer-fragments] ERROR: {exc}", flush=True)
        return 2
    print(json.dumps(result["source"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
