"""Pure validation primitives for the locked DepMap 26Q3 time-out analysis."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np


class TemporalValidationError(RuntimeError):
    """Raised when temporal evidence violates the frozen analysis contract."""


@dataclass(frozen=True)
class AlignedTemporalCohort:
    gene_ids: np.ndarray
    canonical_gene_indices: np.ndarray
    component_ids: np.ndarray
    observed_gene_mean: np.ndarray
    primary_prediction: np.ndarray
    selected_baseline_prediction: np.ndarray
    global_prediction: np.ndarray
    new_model_ids: np.ndarray
    finite_model_fraction: np.ndarray


def hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _one_dimensional_unique_strings(values: Sequence[Any], *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise TemporalValidationError(f"{label} must be a non-empty one-dimensional array")
    strings_list: list[str] = []
    for value in array.tolist():
        if value is None or (
            isinstance(value, (float, np.floating)) and not math.isfinite(float(value))
        ):
            raise TemporalValidationError(f"{label} contains a missing identifier")
        exact = str(value)
        if not exact or exact != exact.strip():
            raise TemporalValidationError(
                f"{label} contains an empty or whitespace-normalized identifier"
            )
        strings_list.append(exact)
    strings = np.asarray(strings_list)
    if len(set(strings.tolist())) != strings.size:
        raise TemporalValidationError(f"{label} contains duplicate identifiers")
    return strings


def parse_gene_headers(
    headers: Sequence[Any], pattern: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse only exact ``SYMBOL (ENTREZ_ID)`` columns and reject symbol aliases."""
    compiled = re.compile(pattern)
    symbols: list[str] = []
    entrez_ids: list[str] = []
    positions: list[int] = []
    for position, raw in enumerate(headers):
        header = str(raw)
        match = compiled.fullmatch(header)
        if match is None:
            continue
        symbol = match.group("symbol")
        entrez = match.group("entrez")
        if symbol != symbol.strip():
            raise TemporalValidationError(
                f"matched gene symbol has boundary whitespace: {header!r}"
            )
        if not symbol or not entrez:
            raise TemporalValidationError(f"invalid matched gene header: {header!r}")
        symbols.append(symbol)
        entrez_ids.append(entrez)
        positions.append(position)
    if not symbols:
        raise TemporalValidationError("no 26Q3 gene columns match the locked header regex")
    if len(set(symbols)) != len(symbols):
        duplicates = sorted(
            value for value, count in Counter(symbols).items() if count > 1
        )
        raise TemporalValidationError(
            f"duplicate exact symbols after terminal Entrez removal: {duplicates[:10]!r}"
        )
    return (
        np.asarray(symbols),
        np.asarray(entrez_ids),
        np.asarray(positions, dtype=np.int64),
    )


def select_new_models(
    model_ids_26q3: Sequence[Any],
    model_ids_24q4: Sequence[Any],
    *,
    minimum_new_models: int,
) -> tuple[np.ndarray, np.ndarray]:
    current = _one_dimensional_unique_strings(model_ids_26q3, label="26Q3 ModelIDs")
    previous = _one_dimensional_unique_strings(model_ids_24q4, label="24Q4 ModelIDs")
    previous_set = set(previous.tolist())
    mask = np.asarray([value not in previous_set for value in current.tolist()], dtype=bool)
    selected = current[mask]
    if selected.size < int(minimum_new_models):
        raise TemporalValidationError(
            f"new-model cohort has {selected.size} models; locked minimum is {minimum_new_models}"
        )
    return selected, np.flatnonzero(mask)


def audit_organoid_overlap(
    temporal_model_ids: Sequence[Any], organoid_model_ids: Sequence[Any]
) -> dict[str, Any]:
    """Audit exact ModelID overlap without aliases or normalization."""
    temporal = _one_dimensional_unique_strings(
        temporal_model_ids, label="temporal new-model ModelIDs"
    )
    organoid = _one_dimensional_unique_strings(
        organoid_model_ids, label="organoid-source ModelIDs"
    )
    organoid_set = set(organoid.tolist())
    overlap_mask = np.asarray(
        [model_id in organoid_set for model_id in temporal.tolist()], dtype=bool
    )
    overlap_ids = temporal[overlap_mask]
    nonoverlap_ids = temporal[~overlap_mask]
    resolved = bool(nonoverlap_ids.size == 0)
    return {
        "audit_status": (
            "resolved_exact_model_id_audit"
            if resolved
            else "unresolved_identifier_namespace_mismatch"
        ),
        "n_temporal_models": int(temporal.size),
        "n_organoid_source_models": int(organoid.size),
        "resolution_basis": (
            "all_temporal_model_ids_exactly_present_in_organoid_source"
            if resolved
            else "unmatched_model_ids_require_authoritative_frozen_crosswalk"
        ),
        "raw_exact_string_intersection_count": int(overlap_ids.size),
        "n_overlap_models": int(overlap_ids.size) if resolved else None,
        "overlap_model_ids": overlap_ids,
        "n_nonoverlap_temporal_models": int(nonoverlap_ids.size),
        "nonoverlap_temporal_model_ids": nonoverlap_ids,
        "nonoverlap_temporal_positions": np.flatnonzero(~overlap_mask),
        "independence_claim_forbidden": bool(not resolved or overlap_ids.size > 0),
    }


def align_temporal_cohort(
    *,
    outcome_model_by_gene: np.ndarray,
    outcome_gene_symbols: Sequence[Any],
    new_model_ids: Sequence[Any],
    consensus_gene_ids: Sequence[Any],
    consensus_canonical_gene_indices: Sequence[Any],
    consensus_component_ids: Sequence[Any],
    primary_prediction: np.ndarray,
    selected_baseline_prediction: np.ndarray,
    global_prediction: np.ndarray,
    minimum_finite_fraction: float,
    minimum_eligible_genes: int,
) -> AlignedTemporalCohort:
    outcomes = np.asarray(outcome_model_by_gene, dtype=np.float64)
    symbols = _one_dimensional_unique_strings(outcome_gene_symbols, label="26Q3 gene symbols")
    models = _one_dimensional_unique_strings(new_model_ids, label="new-model ModelIDs")
    genes = _one_dimensional_unique_strings(consensus_gene_ids, label="consensus gene IDs")
    canonical_indices = np.asarray(consensus_canonical_gene_indices)
    components = np.asarray(consensus_component_ids)
    if outcomes.ndim != 2 or outcomes.shape != (models.size, symbols.size):
        raise TemporalValidationError("26Q3 outcome matrix does not align to models and symbols")
    if components.ndim != 1 or components.size != genes.size:
        raise TemporalValidationError("consensus component IDs do not align to genes")
    if (
        canonical_indices.ndim != 1
        or canonical_indices.size != genes.size
        or canonical_indices.dtype.kind not in "iu"
        or np.any(canonical_indices < 0)
        or len(set(canonical_indices.tolist())) != canonical_indices.size
    ):
        raise TemporalValidationError(
            "consensus canonical gene indices must be unique nonnegative integers aligned to genes"
        )
    component_values = [str(value) for value in components.tolist()]
    if any(not value or value != value.strip() for value in component_values):
        raise TemporalValidationError(
            "consensus component IDs contain an empty or whitespace-normalized value"
        )
    component_strings = np.asarray(component_values)
    if not 0.0 <= float(minimum_finite_fraction) <= 1.0:
        raise TemporalValidationError("minimum finite fraction must be between zero and one")
    if int(minimum_eligible_genes) < 0:
        raise TemporalValidationError("minimum eligible genes cannot be negative")
    prediction_vectors = {
        "primary": np.asarray(primary_prediction, dtype=np.float64),
        "selected baseline": np.asarray(selected_baseline_prediction, dtype=np.float64),
        "global": np.asarray(global_prediction, dtype=np.float64),
    }
    for label, vector in prediction_vectors.items():
        if vector.ndim != 1 or vector.size != genes.size or not np.all(np.isfinite(vector)):
            raise TemporalValidationError(f"{label} prediction is not a finite consensus-aligned vector")
    symbol_to_column = {value: index for index, value in enumerate(symbols.tolist())}
    consensus_positions: list[int] = []
    outcome_positions: list[int] = []
    for index, gene in enumerate(genes.tolist()):
        column = symbol_to_column.get(gene)
        if column is not None:
            consensus_positions.append(index)
            outcome_positions.append(column)
    if not consensus_positions:
        raise TemporalValidationError("26Q3 has no exact-symbol overlap with frozen consensus")
    consensus_index = np.asarray(consensus_positions, dtype=np.int64)
    outcome_index = np.asarray(outcome_positions, dtype=np.int64)
    aligned = outcomes[:, outcome_index]
    finite_fraction = np.mean(np.isfinite(aligned), axis=0)
    eligible = finite_fraction >= float(minimum_finite_fraction)
    if int(np.sum(eligible)) < int(minimum_eligible_genes):
        raise TemporalValidationError(
            f"eligible exact-symbol genes={int(np.sum(eligible))}; locked minimum={minimum_eligible_genes}"
        )
    consensus_index = consensus_index[eligible]
    aligned = aligned[:, eligible]
    finite_fraction = finite_fraction[eligible]
    counts = np.sum(np.isfinite(aligned), axis=0)
    sums = np.sum(np.where(np.isfinite(aligned), aligned, 0.0), axis=0)
    observed = sums / counts
    if not np.all(np.isfinite(observed)):
        raise TemporalValidationError("eligible temporal gene means are not finite")
    return AlignedTemporalCohort(
        gene_ids=genes[consensus_index],
        canonical_gene_indices=canonical_indices[consensus_index],
        component_ids=component_strings[consensus_index],
        observed_gene_mean=observed,
        primary_prediction=prediction_vectors["primary"][consensus_index],
        selected_baseline_prediction=prediction_vectors["selected baseline"][consensus_index],
        global_prediction=prediction_vectors["global"][consensus_index],
        new_model_ids=models,
        finite_model_fraction=finite_fraction,
    )


def _correlation(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float | None:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if rank:
        left = _average_ranks(left)
        right = _average_ranks(right)
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    if denominator <= 0.0:
        return None
    return float(np.dot(left, right) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def regression_metrics(
    prediction: np.ndarray, target: np.ndarray, global_prediction: np.ndarray
) -> dict[str, float | None]:
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    reference = np.asarray(global_prediction, dtype=np.float64)
    if predicted.shape != observed.shape or reference.shape != observed.shape or observed.ndim != 1:
        raise TemporalValidationError("metric vectors must be aligned one-dimensional arrays")
    if observed.size < 2 or not all(np.all(np.isfinite(value)) for value in (predicted, observed, reference)):
        raise TemporalValidationError("metric vectors must contain at least two finite values")
    error = predicted - observed
    reference_error = reference - observed
    sse = float(np.dot(error, error))
    reference_sse = float(np.dot(reference_error, reference_error))
    return {
        "n_genes": int(observed.size),
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "pearson": _correlation(predicted, observed),
        "spearman": _correlation(predicted, observed, rank=True),
        "r2_vs_frozen_consensus_global": (
            None if reference_sse <= 0.0 else float(1.0 - sse / reference_sse)
        ),
        "sse": sse,
        "reference_sse": reference_sse,
    }


def component_block_bootstrap(
    cohort: AlignedTemporalCohort,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0 or not 0.0 < confidence_level < 1.0:
        raise TemporalValidationError("invalid bootstrap contract")
    groups: dict[str, list[int]] = {}
    for index, component in enumerate(cohort.component_ids.tolist()):
        groups.setdefault(str(component), []).append(index)
    if len(groups) < 2:
        raise TemporalValidationError("component bootstrap requires at least two components")
    blocks = [np.asarray(indices, dtype=np.int64) for _, indices in sorted(groups.items())]
    rng = np.random.default_rng(int(seed))
    delta = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[int(value)] for value in sampled])
        target = cohort.observed_gene_mean[indices]
        reference = cohort.global_prediction[indices]
        denominator = float(np.sum(np.square(reference - target)))
        if denominator <= 0.0:
            delta[replicate] = np.nan
            continue
        primary_sse = float(np.sum(np.square(cohort.primary_prediction[indices] - target)))
        baseline_sse = float(
            np.sum(np.square(cohort.selected_baseline_prediction[indices] - target))
        )
        delta[replicate] = (baseline_sse - primary_sse) / denominator
    finite = delta[np.isfinite(delta)]
    if finite.size != replicates:
        raise TemporalValidationError("component bootstrap produced undefined replicates")
    alpha = (1.0 - float(confidence_level)) / 2.0
    return {
        "unit": "sequence_component",
        "replicates": int(replicates),
        "confidence_level": float(confidence_level),
        "seed": int(seed),
        "n_components": len(blocks),
        "delta_r2_vs_frozen_selected_baseline": {
            "ci_low": float(np.quantile(finite, alpha)),
            "ci_high": float(np.quantile(finite, 1.0 - alpha)),
            "replicate_mean": float(np.mean(finite)),
        },
    }


def evaluate_temporal_cohort(
    cohort: AlignedTemporalCohort,
    *,
    bootstrap_contract: Mapping[str, Any],
    gate_contract: Mapping[str, Any],
) -> dict[str, Any]:
    primary = regression_metrics(
        cohort.primary_prediction, cohort.observed_gene_mean, cohort.global_prediction
    )
    baseline = regression_metrics(
        cohort.selected_baseline_prediction,
        cohort.observed_gene_mean,
        cohort.global_prediction,
    )
    primary_r2 = primary["r2_vs_frozen_consensus_global"]
    baseline_r2 = baseline["r2_vs_frozen_consensus_global"]
    if primary_r2 is None or baseline_r2 is None:
        raise TemporalValidationError("temporal R-squared reference denominator is zero")
    delta_r2 = float(primary_r2 - baseline_r2)
    bootstrap = component_block_bootstrap(
        cohort,
        replicates=int(bootstrap_contract["replicates"]),
        confidence_level=float(bootstrap_contract["confidence_level"]),
        seed=int(bootstrap_contract["seed"]),
    )
    ci_low = bootstrap["delta_r2_vs_frozen_selected_baseline"]["ci_low"]
    gates = {
        "positive_gene_mean_r2": primary_r2
        > float(gate_contract["gene_mean_r2_min_exclusive"]),
        "minimum_delta_r2": delta_r2
        >= float(gate_contract["delta_r2_vs_frozen_selected_baseline_min"]),
        "positive_delta_r2_bootstrap_lower": ci_low
        > float(gate_contract["delta_r2_bootstrap_lower_min_exclusive"]),
    }
    return {
        "cohort": {
            "n_new_models": int(cohort.new_model_ids.size),
            "n_eligible_genes": int(cohort.gene_ids.size),
            "n_sequence_components": int(len(set(cohort.component_ids.tolist()))),
            "minimum_finite_model_fraction": float(np.min(cohort.finite_model_fraction)),
        },
        "primary_method": primary,
        "selected_baseline": baseline,
        "delta_r2_vs_frozen_selected_baseline": delta_r2,
        "component_block_bootstrap": bootstrap,
        "gates": {**gates, "all_passed": bool(all(gates.values()))},
    }


def evaluate_shared_model_pipeline_correction(
    *,
    primary_cohort: AlignedTemporalCohort,
    outcome_model_ids: Sequence[Any],
    outcome_gene_symbols: Sequence[Any],
    outcome_model_by_gene: np.ndarray,
    baseline_model_ids: Sequence[Any],
    baseline_gene_symbols: Sequence[Any],
    baseline_gene_by_model: np.ndarray,
    minimum_finite_paired_shared_models_per_gene: int,
    minimum_eligible_genes: int,
    bootstrap_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Quantify the registered shared-model processing shift as a secondary analysis."""
    current_models = _one_dimensional_unique_strings(
        outcome_model_ids, label="26Q3 ModelIDs"
    )
    previous_models = _one_dimensional_unique_strings(
        baseline_model_ids, label="24Q4 ModelIDs"
    )
    current_genes = _one_dimensional_unique_strings(
        outcome_gene_symbols, label="26Q3 gene symbols"
    )
    previous_genes = _one_dimensional_unique_strings(
        baseline_gene_symbols, label="24Q4 gene symbols"
    )
    current_values = np.asarray(outcome_model_by_gene)
    previous_values = np.asarray(baseline_gene_by_model)
    if (
        current_values.ndim != 2
        or current_values.shape != (current_models.size, current_genes.size)
        or current_values.dtype.kind not in "fiu"
    ):
        raise TemporalValidationError("26Q3 shared-model matrix shape or dtype is invalid")
    if (
        previous_values.ndim != 2
        or previous_values.shape != (previous_genes.size, previous_models.size)
        or previous_values.dtype.kind not in "fiu"
    ):
        raise TemporalValidationError("24Q4 gene-effect matrix shape or dtype is invalid")
    if int(minimum_finite_paired_shared_models_per_gene) < 1 or int(minimum_eligible_genes) < 1:
        raise TemporalValidationError("shared-model correction eligibility minima must be positive")

    previous_model_positions = {value: index for index, value in enumerate(previous_models.tolist())}
    expected_new_ids = current_models[
        np.asarray([value not in previous_model_positions for value in current_models.tolist()])
    ]
    if not np.array_equal(
        _one_dimensional_unique_strings(primary_cohort.new_model_ids, label="primary new-model ModelIDs"),
        expected_new_ids,
    ):
        raise TemporalValidationError("correction new-model cohort differs from primary")
    shared_current_positions = np.asarray(
        [index for index, value in enumerate(current_models.tolist()) if value in previous_model_positions],
        dtype=np.int64,
    )
    shared_ids = current_models[shared_current_positions]
    shared_previous_positions = np.asarray(
        [previous_model_positions[value] for value in shared_ids.tolist()], dtype=np.int64
    )
    evidence = {
        "shared_model_ids": shared_ids,
        "shared_26q3_source_row_indices": shared_current_positions,
        "shared_24q4_source_column_indices": shared_previous_positions,
    }
    report: dict[str, Any] = {
        "role": "secondary_cannot_replace_new_model_primary",
        "evaluation_cohort": "same_new_models_as_primary",
        "prediction_source": "frozen_24q4_consensus_predictions_plus_per_gene_correction",
        "reference": "frozen_24q4_consensus_global_prediction_plus_same_correction",
        "correction_source": "paired_shared_model_26q3_minus_24q4_gene_effect_median",
        "n_shared_models": int(shared_ids.size),
        "n_primary_eligible_genes": int(primary_cohort.gene_ids.size),
        "minimum_finite_paired_shared_models_per_gene": int(
            minimum_finite_paired_shared_models_per_gene
        ),
        "minimum_eligible_genes": int(minimum_eligible_genes),
        "primary_temporal_gates_unchanged": True,
    }
    if shared_ids.size < minimum_finite_paired_shared_models_per_gene:
        report.update({
            "status": "ineligible_insufficient_shared_models",
            "n_eligible_genes": 0,
        })
        return report, evidence

    primary_ids = _one_dimensional_unique_strings(
        primary_cohort.gene_ids, label="primary temporal gene IDs"
    )
    canonical = np.asarray(primary_cohort.canonical_gene_indices)
    if (
        canonical.ndim != 1
        or canonical.shape != primary_ids.shape
        or canonical.dtype.kind not in "iu"
        or np.any(canonical < 0)
        or np.any(canonical >= previous_genes.size)
        or not np.array_equal(previous_genes[canonical], primary_ids)
    ):
        raise TemporalValidationError(
            "primary canonical gene indices differ from locked 24Q4 gene manifest"
        )
    current_gene_positions = {value: index for index, value in enumerate(current_genes.tolist())}
    try:
        aligned_current_gene_positions = np.asarray(
            [current_gene_positions[value] for value in primary_ids.tolist()], dtype=np.int64
        )
    except KeyError as exc:
        raise TemporalValidationError(
            f"primary gene absent from exact 26Q3 symbol mapping: {exc}"
        ) from exc
    current_shared = np.asarray(
        current_values[np.ix_(shared_current_positions, aligned_current_gene_positions)],
        dtype=np.float64,
    )
    previous_shared = np.asarray(
        previous_values[np.ix_(canonical, shared_previous_positions)].T,
        dtype=np.float64,
    )
    finite_pairs = np.isfinite(current_shared) & np.isfinite(previous_shared)
    paired_counts = np.sum(finite_pairs, axis=0)
    eligible = paired_counts >= int(minimum_finite_paired_shared_models_per_gene)
    evidence["shared_correction_paired_counts_per_primary_gene"] = paired_counts.astype(np.int64)
    evidence["shared_correction_eligible_gene_ids"] = primary_ids[eligible]
    report["n_eligible_genes"] = int(np.sum(eligible))
    if report["n_eligible_genes"] < minimum_eligible_genes:
        report["status"] = "ineligible_insufficient_paired_genes"
        return report, evidence

    eligible_pairs = finite_pairs[:, eligible]
    differences = current_shared[:, eligible] - previous_shared[:, eligible]
    corrections = np.asarray(
        [np.median(differences[eligible_pairs[:, index], index])
         for index in range(int(np.sum(eligible)))],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(corrections)):
        raise TemporalValidationError("shared-model median corrections are not finite")
    corrected = replace(
        primary_cohort,
        gene_ids=primary_cohort.gene_ids[eligible],
        canonical_gene_indices=primary_cohort.canonical_gene_indices[eligible],
        component_ids=primary_cohort.component_ids[eligible],
        observed_gene_mean=primary_cohort.observed_gene_mean[eligible],
        primary_prediction=primary_cohort.primary_prediction[eligible] + corrections,
        selected_baseline_prediction=(
            primary_cohort.selected_baseline_prediction[eligible] + corrections
        ),
        global_prediction=primary_cohort.global_prediction[eligible] + corrections,
        finite_model_fraction=primary_cohort.finite_model_fraction[eligible],
    )
    primary = regression_metrics(
        corrected.primary_prediction, corrected.observed_gene_mean, corrected.global_prediction
    )
    baseline = regression_metrics(
        corrected.selected_baseline_prediction,
        corrected.observed_gene_mean,
        corrected.global_prediction,
    )
    primary_r2 = primary["r2_vs_frozen_consensus_global"]
    baseline_r2 = baseline["r2_vs_frozen_consensus_global"]
    primary["r2_vs_corrected_consensus_global"] = primary.pop(
        "r2_vs_frozen_consensus_global"
    )
    baseline["r2_vs_corrected_consensus_global"] = baseline.pop(
        "r2_vs_frozen_consensus_global"
    )
    report.update({
        "status": (
            "complete" if primary_r2 is not None and baseline_r2 is not None
            else "undefined_corrected_reference_denominator"
        ),
        "primary_method": primary,
        "selected_baseline": baseline,
        "delta_r2_vs_corrected_selected_baseline": (
            None if primary_r2 is None or baseline_r2 is None
            else float(primary_r2 - baseline_r2)
        ),
    })
    try:
        bootstrap = component_block_bootstrap(
            corrected,
            replicates=int(bootstrap_contract["replicates"]),
            confidence_level=float(bootstrap_contract["confidence_level"]),
            seed=int(bootstrap_contract["seed"]),
        )
        bootstrap["delta_r2_vs_corrected_selected_baseline"] = bootstrap.pop(
            "delta_r2_vs_frozen_selected_baseline"
        )
        report["component_block_bootstrap"] = bootstrap
    except TemporalValidationError as exc:
        report["component_block_bootstrap"] = {"status": "undefined", "reason": str(exc)}
    evidence.update({
        "shared_correction_values": corrections,
        "shared_correction_observed_new_model_gene_mean": corrected.observed_gene_mean,
        "shared_correction_primary_prediction": corrected.primary_prediction,
        "shared_correction_selected_baseline_prediction": corrected.selected_baseline_prediction,
        "shared_correction_global_prediction": corrected.global_prediction,
    })
    return report, evidence
