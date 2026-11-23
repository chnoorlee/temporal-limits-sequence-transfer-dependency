"""Transparent post-access compatibility runner for the DepMap 26Q3 index header.

The pre-access validator required the first CSV header field to be ``ModelID``.
The official 26Q3 file uses an empty first field.  This runner preserves the
frozen validator files and changes only that input-schema interpretation.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

import run_temporal_26q3_validation as frozen


AMENDMENT_ID = "depmap-26q3-empty-index-header-compat-v1"
AMENDMENT_SCHEMA = "DepPLM-temporal-post-access-schema-amendment-v1"
RESULT_SCHEMA = "DepPLM-temporal-post-access-schema-amendment-result-v1"
TRIGGER_ERROR = (
    "26Q3 CSV first column must be exactly 'ModelID' and include gene columns"
)


def _read_temporal_csv_empty_index_header(
    path: Path, *, gene_header_pattern: str
) -> tuple[pd.DataFrame, Any, Any, Any, list[str]]:
    """Read the official blank index header without changing outcome bytes."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_header = next(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        raise frozen.TemporalValidationError(
            f"cannot read 26Q3 CSV header: {exc}"
        ) from exc
    if len(raw_header) < 2 or raw_header[0] != "":
        raise frozen.TemporalValidationError(
            "post-access schema amendment requires an empty first CSV header field"
        )
    if len(set(raw_header)) != len(raw_header):
        raise frozen.TemporalValidationError(
            "26Q3 CSV contains duplicate raw column headers"
        )
    symbols, entrez_ids, gene_positions = frozen.parse_gene_headers(
        raw_header[1:], gene_header_pattern
    )
    try:
        frame = pd.read_csv(
            path,
            index_col=0,
            encoding="utf-8-sig",
            low_memory=False,
        )
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise frozen.TemporalValidationError(
            f"cannot parse 26Q3 outcome CSV: {exc}"
        ) from exc
    if frame.index.name is not None or frame.columns.tolist() != raw_header[1:]:
        raise frozen.TemporalValidationError(
            "parsed 26Q3 table does not preserve the official blank index header"
        )
    return frame, symbols, entrez_ids, gene_positions, raw_header


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(frozen._json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _declare_amendment(args: Any) -> tuple[Path, Path]:
    if frozen.IDENTIFIER.fullmatch(args.analysis_id) is None:
        raise frozen.TemporalValidationError("invalid analysis identifier")
    freeze_path = args.freeze_manifest.resolve(strict=True)
    outcome_path = args.outcome.resolve(strict=True)
    lock = frozen._load_lock(args.lock)
    ledger = frozen.ROOT / lock["temporal_26q3"]["access_ledger_path"]
    events = frozen._read_ledger(ledger)
    failures = [
        event
        for event in events
        if event.get("event") == "validation_failed"
        and event.get("error_message") == TRIGGER_ERROR
    ]
    if not failures:
        raise frozen.TemporalValidationError(
            "schema amendment requires the recorded frozen-parser failure"
        )
    with outcome_path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_header = next(csv.reader(handle))
    if len(raw_header) < 2 or raw_header[0] != "":
        raise frozen.TemporalValidationError(
            "schema amendment applies only to the official empty index header"
        )
    freeze = frozen._load_json(freeze_path, label="temporal freeze")
    wrapper_record = frozen._file_record(Path(__file__))
    plan_path = (
        frozen.ROOT
        / "results/family_inductive_v2/temporal-amendments"
        / f"{args.analysis_id}-{AMENDMENT_ID}.json"
    )
    payload = {
        "schema": AMENDMENT_SCHEMA,
        "status": "declared_after_outcome_access_before_schema_compatibility_retry",
        "amendment_id": AMENDMENT_ID,
        "analysis_id": args.analysis_id,
        "created_at_utc": frozen._utc_now(),
        "triggering_failure": failures[-1],
        "pre_access_freeze": frozen._file_record(freeze_path),
        "frozen_validator_runner": freeze["frozen_files"]["validator_runner"],
        "frozen_validator_library": freeze["frozen_files"]["validator_library"],
        "compatibility_runner": wrapper_record,
        "source_outcome": frozen._file_record(outcome_path),
        "observed_schema": {
            "first_header_field": "",
            "column_count": len(raw_header),
            "interpretation": "first field is the ModelID row-index label",
        },
        "change_scope": [
            "accept only an empty first CSV header field",
            "preserve the official source bytes and all gene headers",
            "assign no synthetic values and change no numeric data",
        ],
        "unchanged_analysis_contract": [
            "frozen predictions",
            "model and baseline",
            "cohort and gene mapping",
            "endpoints and metrics",
            "thresholds and bootstrap seed",
        ],
        "limitation": (
            "This is a post-outcome-access input-schema correction. The numerical "
            "analysis remains locked, but the compatibility layer was not frozen "
            "before 26Q3 access and must not be described as prospectively frozen."
        ),
    }
    if plan_path.exists():
        existing = frozen._load_json(plan_path, label="schema amendment plan")
        for key in (
            "schema",
            "amendment_id",
            "analysis_id",
            "pre_access_freeze",
            "compatibility_runner",
            "source_outcome",
            "observed_schema",
            "change_scope",
            "unchanged_analysis_contract",
        ):
            if existing.get(key) != payload[key]:
                raise frozen.TemporalValidationError(
                    "existing schema amendment plan differs from this retry"
                )
    else:
        _write_new_json(plan_path, payload)
        frozen._append_ledger(
            ledger,
            {
                "schema": "DepPLM-temporal-access-event-v1",
                "event": "post_access_schema_amendment_declared_before_retry",
                "recorded_at_utc": frozen._utc_now(),
                "analysis_id": args.analysis_id,
                "amendment_id": AMENDMENT_ID,
                "amendment_plan": frozen._file_record(plan_path),
                "outcome_sha256": payload["source_outcome"]["sha256"],
            },
        )
    return plan_path, ledger


def main(argv: Sequence[str] | None = None) -> int:
    args = frozen.parse_args(argv)
    if args.command != "validate":
        print(
            "[temporal-26q3] ERROR: schema compatibility runner supports validate only",
            file=sys.stderr,
            flush=True,
        )
        return 2
    try:
        plan_path, ledger = _declare_amendment(args)
        frozen._read_temporal_csv = _read_temporal_csv_empty_index_header
        destination = frozen.run_temporal_validation(
            analysis_id=args.analysis_id,
            freeze_manifest=args.freeze_manifest,
            outcome_path=args.outcome,
            output_root=args.output_root,
            lock_path=args.lock,
        )
        result_path = plan_path.with_name(
            f"{args.analysis_id}-{AMENDMENT_ID}-result.json"
        )
        result = {
            "schema": RESULT_SCHEMA,
            "status": "validation_completed_with_post_access_schema_amendment",
            "recorded_at_utc": frozen._utc_now(),
            "amendment_plan": frozen._file_record(plan_path),
            "validation_completion": frozen._file_record(
                destination / "completion.json"
            ),
        }
        _write_new_json(result_path, result)
        frozen._append_ledger(
            ledger,
            {
                "schema": "DepPLM-temporal-access-event-v1",
                "event": "post_access_schema_amendment_result_bound",
                "recorded_at_utc": frozen._utc_now(),
                "analysis_id": args.analysis_id,
                "amendment_id": AMENDMENT_ID,
                "amendment_result": frozen._file_record(result_path),
            },
        )
        print(destination)
        return 0
    except (OSError, ValueError, frozen.TemporalValidationError) as exc:
        print(f"[temporal-26q3] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
