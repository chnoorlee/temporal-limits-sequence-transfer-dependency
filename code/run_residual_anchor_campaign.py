"""Execute the fixed 36-run residual-anchor research campaign.

The campaign is intentionally separate from corrected-v2. It compares absolute
and training-gene-mean-residual targets for two architectures under identical
protocols, seeds, inputs, and outer splits. Every completion is replay-validated
before the next matrix cell is accepted.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from aggregate_corrected_campaign import (
    CANONICAL_N_CELLS,
    CANONICAL_N_GENES,
    CampaignSpec,
    RunKey,
    _load_gene_effect,
)
from aggregate_residual_anchor_campaign import (
    ResidualAnchorValidationError,
    completion_directories,
    residual_anchor_spec,
    validate_research_run,
)
from embedding_provenance import validate_embedding_bundle
from run_corrected_campaign import (
    CampaignExecutionError,
    SourceSnapshot,
    _acquire_lock,
    _append_jsonl,
    _release_lock,
    _run_and_tee,
    _sha256_file,
    _utc_now,
)


ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results" / "residual_anchor_v1"
SOURCE_PATHS = (
    "code/run_corrected_experiments.py",
    "code/corrected_protocol.py",
    "code/models.py",
    "code/03_embed_proteins.py",
    "code/embedding_provenance.py",
    "code/aggregate_corrected_campaign.py",
    "code/aggregate_residual_anchor_campaign.py",
    "code/run_residual_anchor_campaign.py",
)
RUNTIME_INPUTS = {
    "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv": (
        506628654,
        "2a71dc94110efcc0221eae821bb93a9f03b54bea16f005818911a09d33383d56",
    ),
    "data/depmap/OmicsSomaticMutations.csv": (
        338945382,
        "92e86ab6a8457ef99908daf2a2c82d9529320e23cd6c6b7adc128adc1e0f41c8",
    ),
}
CONDITION_ORDER = (
    "RA-PLM-absolute",
    "RA-PLM-anchored",
    "RA-DepPLM-absolute",
    "RA-DepPLM-anchored",
)
PROTOCOL_ORDER = ("p1", "transfer-forward", "transfer-reverse")


def capture_source_snapshot(workspace_root: Path) -> SourceSnapshot:
    records: dict[str, dict[str, int | str]] = {}
    for relative in SOURCE_PATHS:
        path = workspace_root / relative
        if not path.is_file():
            raise CampaignExecutionError(f"missing research source file: {path}")
        stat = path.stat()
        records[relative] = {
            "bytes": int(stat.st_size),
            "sha256": _sha256_file(path),
        }
    return SourceSnapshot(records=records)


def validate_runtime_inputs(workspace_root: Path) -> None:
    for relative, (expected_bytes, expected_sha256) in RUNTIME_INPUTS.items():
        path = workspace_root / relative
        if not path.is_file():
            raise CampaignExecutionError(f"missing pinned research input: {path}")
        if path.stat().st_size != expected_bytes:
            raise CampaignExecutionError(f"pinned research input has wrong size: {path}")
        if _sha256_file(path) != expected_sha256:
            raise CampaignExecutionError(f"pinned research input has wrong SHA-256: {path}")


def assert_source_snapshot_unchanged(
    workspace_root: Path, expected: SourceSnapshot
) -> None:
    current = capture_source_snapshot(workspace_root)
    if current != expected:
        changed = sorted(
            relative
            for relative in set(expected.records) | set(current.records)
            if expected.records.get(relative) != current.records.get(relative)
        )
        raise CampaignExecutionError(
            "research source changed during the campaign: " + ", ".join(changed)
        )


def ordered_campaign_keys(spec: CampaignSpec) -> tuple[RunKey, ...]:
    keys = tuple(
        RunKey(protocol, condition, int(seed))
        for protocol in PROTOCOL_ORDER
        for condition in CONDITION_ORDER
        for seed in spec.seeds
    )
    expected = set(spec.expected_keys())
    if len(keys) != len(expected) or set(keys) != expected:
        raise CampaignExecutionError(
            "residual-anchor ordering does not match the fixed 36-run matrix"
        )
    return keys


def _validate_existing(
    directories: Sequence[Path],
    *,
    workspace_root: Path,
    spec: CampaignSpec,
    gene_effect: Any,
    embedding_provenance: Mapping[str, Any],
    reference_hash_cache: dict[tuple[int, ...], str],
) -> dict[RunKey, Any]:
    indexed: dict[RunKey, Any] = {}
    for directory in directories:
        run = validate_research_run(
            directory,
            workspace_root=workspace_root,
            spec=spec,
            gene_effect=gene_effect,
            embedding_provenance=embedding_provenance,
            reference_hash_cache=reference_hash_cache,
        )
        if run.key in indexed:
            raise CampaignExecutionError(
                f"duplicate completed research cell {run.key.label()}: "
                f"{indexed[run.key].path} and {run.path}"
            )
        indexed[run.key] = run
    return indexed


def run_campaign(
    *,
    workspace_root: Path,
    results_root: Path,
    python_executable: Path,
    device: str,
    dry_run: bool,
) -> int:
    workspace_root = workspace_root.resolve()
    results_root = results_root.resolve()
    expected_results_root = (
        workspace_root / "results" / "residual_anchor_v1"
    ).resolve()
    if results_root != expected_results_root:
        raise CampaignExecutionError(
            "the residual-anchor runner has a fixed output root; --results-root must be "
            f"{expected_results_root}"
        )
    if not python_executable.is_file():
        raise CampaignExecutionError(
            f"Python executable does not exist: {python_executable}"
        )

    validate_runtime_inputs(workspace_root)
    spec = residual_anchor_spec()
    ordered_keys = ordered_campaign_keys(spec)
    source_snapshot = capture_source_snapshot(workspace_root)
    embedding_provenance = validate_embedding_bundle(workspace_root)
    gene_effect = _load_gene_effect(
        workspace_root,
        n_genes=CANONICAL_N_GENES,
        n_cells=CANONICAL_N_CELLS,
    )
    reference_hash_cache: dict[tuple[int, ...], str] = {}
    completed = _validate_existing(
        completion_directories(results_root),
        workspace_root=workspace_root,
        spec=spec,
        gene_effect=gene_effect,
        embedding_provenance=embedding_provenance,
        reference_hash_cache=reference_hash_cache,
    )
    unexpected = sorted(set(completed) - set(ordered_keys))
    if unexpected:
        raise CampaignExecutionError(
            "research output root contains unexpected completed cells: "
            + ", ".join(key.label() for key in unexpected)
        )
    pending = [key for key in ordered_keys if key not in completed]
    print(
        f"[residual-anchor] validated={len(completed)} "
        f"pending={len(pending)} expected={len(ordered_keys)}",
        flush=True,
    )
    if dry_run:
        for key in pending:
            print(key.label())
        return 0

    lock_path = results_root / ".residual-anchor-campaign.lock"
    ledger_path = results_root / "campaign-execution.jsonl"
    log_root = results_root / "execution-logs"
    lock_descriptor = _acquire_lock(
        lock_path,
        {
            "schema": "DepPLM-residual-anchor-v1-campaign-lock",
            "created_utc": _utc_now(),
            "pid": os.getpid(),
            "host": platform.node(),
            "python": str(python_executable),
            "device": device,
            "source_snapshot": source_snapshot.records,
        },
    )
    try:
        for position, key in enumerate(pending, start=len(completed) + 1):
            assert_source_snapshot_unchanged(workspace_root, source_snapshot)
            before = set(completion_directories(results_root))
            started = time.time()
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            log_path = log_root / (
                f"{stamp}-{key.protocol}-{key.condition}-seed-{key.seed}.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if log_path.exists():
                raise CampaignExecutionError(f"refusing to overwrite log: {log_path}")
            command = [
                str(python_executable),
                str(workspace_root / "code" / "run_corrected_experiments.py"),
                "--protocol",
                key.protocol,
                "--condition",
                key.condition,
                "--seed",
                str(key.seed),
                "--device",
                device,
            ]
            _append_jsonl(
                ledger_path,
                {
                    "event": "run_started",
                    "created_utc": _utc_now(),
                    "position": position,
                    "expected_runs": len(ordered_keys),
                    "key": key.label(),
                    "command": command,
                    "log": str(log_path.relative_to(workspace_root)),
                },
            )
            with log_path.open("x", encoding="utf-8", newline="\n") as log_handle:
                return_code = _run_and_tee(
                    command,
                    cwd=workspace_root,
                    log_handle=log_handle,
                )
            assert_source_snapshot_unchanged(workspace_root, source_snapshot)
            if return_code != 0:
                _append_jsonl(
                    ledger_path,
                    {
                        "event": "run_failed",
                        "created_utc": _utc_now(),
                        "key": key.label(),
                        "return_code": return_code,
                        "wall_seconds": time.time() - started,
                    },
                )
                raise CampaignExecutionError(
                    f"research run failed with exit code {return_code}: {key.label()}"
                )

            new_completions = set(completion_directories(results_root)) - before
            if len(new_completions) != 1:
                raise CampaignExecutionError(
                    f"{key.label()} produced {len(new_completions)} new completion "
                    "markers; expected 1"
                )
            directory = next(iter(new_completions))
            validated = validate_research_run(
                directory,
                workspace_root=workspace_root,
                spec=spec,
                gene_effect=gene_effect,
                embedding_provenance=embedding_provenance,
                reference_hash_cache=reference_hash_cache,
            )
            if validated.key != key:
                raise CampaignExecutionError(
                    f"runner produced {validated.key.label()} while "
                    f"{key.label()} was requested"
                )
            if key in completed:
                raise CampaignExecutionError(
                    f"duplicate completed research key: {key.label()}"
                )
            completed[key] = validated
            _append_jsonl(
                ledger_path,
                {
                    "event": "run_validated",
                    "created_utc": _utc_now(),
                    "position": position,
                    "expected_runs": len(ordered_keys),
                    "key": key.label(),
                    "run_directory": str(directory.relative_to(workspace_root)),
                    "wall_seconds": time.time() - started,
                },
            )
            print(
                f"[residual-anchor] validated {position}/{len(ordered_keys)} "
                f"{key.label()}",
                flush=True,
            )

        expected = set(spec.expected_keys())
        if set(completed) != expected:
            missing = sorted(expected - set(completed))
            raise CampaignExecutionError(
                "campaign loop ended with missing research keys: "
                + ", ".join(key.label() for key in missing)
            )
        assert_source_snapshot_unchanged(workspace_root, source_snapshot)
        aggregate_command = [
            str(python_executable),
            str(workspace_root / "code" / "aggregate_residual_anchor_campaign.py"),
            "--results-root",
            str(results_root),
            "--workspace-root",
            str(workspace_root),
        ]
        aggregate_log = log_root / (
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            + "-residual-anchor-aggregate.log"
        )
        with aggregate_log.open("x", encoding="utf-8", newline="\n") as log_handle:
            aggregate_return_code = _run_and_tee(
                aggregate_command,
                cwd=workspace_root,
                log_handle=log_handle,
            )
        assert_source_snapshot_unchanged(workspace_root, source_snapshot)
        if aggregate_return_code != 0:
            raise CampaignExecutionError(
                "independent residual-anchor aggregation failed with exit code "
                f"{aggregate_return_code}"
            )
        _append_jsonl(
            ledger_path,
            {
                "event": "campaign_aggregated",
                "created_utc": _utc_now(),
                "validated_runs": len(completed),
                "aggregate_log": str(aggregate_log.relative_to(workspace_root)),
            },
        )
        return 0
    finally:
        _release_lock(lock_path, lock_descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used for all runs and final aggregation",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_campaign(
            workspace_root=args.workspace_root,
            results_root=args.results_root,
            python_executable=args.python.absolute(),
            device=args.device,
            dry_run=bool(args.dry_run),
        )
    except (
        CampaignExecutionError,
        ResidualAnchorValidationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"[residual-anchor] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
