"""Execute and verify the fixed 99-run corrected DepPLM campaign.

The individual runner owns the scientific protocol.  This module only
orchestrates its immutable matrix, refuses duplicate completed cells, verifies
every new completion against the bound outer-test predictions, and invokes the
independent campaign aggregator after all 99 cells are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

from aggregate_corrected_campaign import (
    CANONICAL_N_CELLS,
    CANONICAL_N_GENES,
    CampaignSpec,
    RunKey,
    RunValidationError,
    _load_gene_effect,
    canonical_spec,
    validate_run_directory,
)
from embedding_provenance import validate_embedding_bundle


ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results" / "corrected_v2"
RUNNER_PATH = ROOT / "code" / "run_corrected_experiments.py"
AGGREGATOR_PATH = ROOT / "code" / "aggregate_corrected_campaign.py"
CORE_SOURCE_PATHS = (
    "code/run_corrected_experiments.py",
    "code/corrected_protocol.py",
    "code/models.py",
    "code/03_embed_proteins.py",
    "code/embedding_provenance.py",
)
PRIORITY_CONDITIONS = (
    "DepPLM",
    "C-DEEPDEP",
    "C-ONEHOT",
    "C-PLM-MLP",
    "C-GCN",
    "C-GCN-comut",
    "A1-ESM",
    "A2-GNN",
    "A3-comut",
    "A4-MT",
    "A5-XATTN",
)
PRIORITY_PROTOCOLS = ("p1", "transfer-forward", "transfer-reverse")


class CampaignExecutionError(RuntimeError):
    """Raised when the campaign cannot continue without weakening its contract."""


@dataclass(frozen=True)
class SourceSnapshot:
    records: Mapping[str, Mapping[str, int | str]]


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def capture_source_snapshot(workspace_root: Path) -> SourceSnapshot:
    """Capture the exact core source bytes that all 99 runs must share."""
    records: dict[str, dict[str, int | str]] = {}
    for relative in CORE_SOURCE_PATHS:
        path = workspace_root / relative
        if not path.is_file():
            raise CampaignExecutionError(f"missing core source file: {path}")
        stat = path.stat()
        records[relative] = {
            "bytes": int(stat.st_size),
            "sha256": _sha256_file(path),
        }
    return SourceSnapshot(records=records)


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
            "core source changed during the canonical campaign: " + ", ".join(changed)
        )


def ordered_campaign_keys(spec: CampaignSpec) -> tuple[RunKey, ...]:
    """Return all canonical keys once, prioritizing DepPLM evidence."""
    condition_order = [
        condition for condition in PRIORITY_CONDITIONS if condition in spec.conditions
    ]
    condition_order.extend(
        condition for condition in spec.conditions if condition not in condition_order
    )
    protocol_order = [
        protocol for protocol in PRIORITY_PROTOCOLS if protocol in spec.protocols
    ]
    protocol_order.extend(
        protocol for protocol in spec.protocols if protocol not in protocol_order
    )
    keys = tuple(
        RunKey(protocol, condition, int(seed))
        for condition in condition_order
        for protocol in protocol_order
        for seed in spec.seeds
    )
    expected = set(spec.expected_keys())
    if len(keys) != len(expected) or set(keys) != expected:
        raise CampaignExecutionError(
            "campaign ordering does not match the canonical matrix"
        )
    return keys


def completion_directories(results_root: Path) -> tuple[Path, ...]:
    """Return only top-level immutable run directories with completion markers."""
    if not results_root.exists():
        return ()
    if not results_root.is_dir():
        raise CampaignExecutionError(f"results root is not a directory: {results_root}")
    return tuple(
        path
        for path in sorted(results_root.iterdir(), key=lambda item: item.name)
        if path.is_dir() and (path / "completion.json").is_file()
    )


def index_validated_runs(runs: Iterable[Any]) -> dict[RunKey, Any]:
    """Index validated runs while refusing ambiguous duplicate matrix cells."""
    indexed: dict[RunKey, Any] = {}
    for run in runs:
        if run.key in indexed:
            raise CampaignExecutionError(
                f"duplicate completed campaign cell {run.key.label()}: "
                f"{indexed[run.key].path} and {run.path}"
            )
        indexed[run.key] = run
    return indexed


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_console(text: str) -> None:
    """Write text without letting a legacy Windows code page abort a campaign."""
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except UnicodeEncodeError:
        console_encoding = sys.stdout.encoding or "utf-8"
        escaped = text.encode(console_encoding, errors="backslashreplace").decode(
            console_encoding
        )
        sys.stdout.write(escaped)
        sys.stdout.flush()


def _run_and_tee(command: Sequence[str], *, cwd: Path, log_handle: TextIO) -> int:
    rendered = subprocess.list2cmdline([str(item) for item in command])
    _write_console(f"[campaign] command={rendered}\n")
    log_handle.write(f"[campaign] command={rendered}\n")
    log_handle.flush()
    child_environment = os.environ.copy()
    child_environment["PYTHONIOENCODING"] = "utf-8"
    child_environment["PYTHONUTF8"] = "1"
    process = subprocess.Popen(
        [str(item) for item in command],
        cwd=cwd,
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is None:
        raise CampaignExecutionError("subprocess stdout pipe was not created")
    for line in process.stdout:
        _write_console(line)
        log_handle.write(line)
        log_handle.flush()
    return int(process.wait())


def _acquire_lock(path: Path, payload: Mapping[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise CampaignExecutionError(
            f"campaign lock already exists; inspect the recorded process before retrying: {path}"
        ) from exc
    try:
        content = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
        os.write(descriptor, content.encode("utf-8"))
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return descriptor


def _release_lock(path: Path, descriptor: int) -> None:
    os.close(descriptor)
    path.unlink(missing_ok=True)


def _validate_existing(
    directories: Sequence[Path],
    *,
    workspace_root: Path,
    spec: CampaignSpec,
    gene_effect: Any,
    embedding_provenance: Mapping[str, Any],
    reference_hash_cache: dict[tuple[int, ...], str],
) -> dict[RunKey, Any]:
    validated = []
    for directory in directories:
        try:
            validated.append(
                validate_run_directory(
                    directory,
                    workspace_root=workspace_root,
                    spec=spec,
                    _gene_effect=gene_effect,
                    _reference_hash_cache=reference_hash_cache,
                    _expected_embedding_provenance=embedding_provenance,
                )
            )
        except RunValidationError as exc:
            raise CampaignExecutionError(
                f"existing completion failed validation [{exc.code}] {directory}: {exc}"
            ) from exc
    return index_validated_runs(validated)


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
    expected_results_root = (workspace_root / "results" / "corrected_v2").resolve()
    if results_root != expected_results_root:
        raise CampaignExecutionError(
            "the canonical runner has a fixed output root; --results-root must be "
            f"{expected_results_root}"
        )
    if not python_executable.is_file():
        raise CampaignExecutionError(
            f"Python executable does not exist: {python_executable}"
        )

    spec = canonical_spec()
    ordered_keys = ordered_campaign_keys(spec)
    source_snapshot = capture_source_snapshot(workspace_root)
    embedding_provenance = validate_embedding_bundle(workspace_root)
    gene_effect = _load_gene_effect(
        workspace_root, n_genes=CANONICAL_N_GENES, n_cells=CANONICAL_N_CELLS
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
    pending = [key for key in ordered_keys if key not in completed]
    print(
        f"[campaign] validated={len(completed)} pending={len(pending)} expected={len(ordered_keys)}",
        flush=True,
    )
    if dry_run:
        for key in pending:
            print(key.label())
        return 0

    lock_path = results_root / ".canonical-campaign.lock"
    ledger_path = results_root / "campaign-execution.jsonl"
    log_root = results_root / "execution-logs"
    lock_descriptor = _acquire_lock(
        lock_path,
        {
            "schema": "DepPLM-corrected-v2.2-campaign-lock",
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
                    command, cwd=workspace_root, log_handle=log_handle
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
                    f"canonical run failed with exit code {return_code}: {key.label()}"
                )
            new_completions = set(completion_directories(results_root)) - before
            if len(new_completions) != 1:
                raise CampaignExecutionError(
                    f"{key.label()} produced {len(new_completions)} new completion markers; expected 1"
                )
            directory = next(iter(new_completions))
            try:
                validated = validate_run_directory(
                    directory,
                    workspace_root=workspace_root,
                    spec=spec,
                    _gene_effect=gene_effect,
                    _reference_hash_cache=reference_hash_cache,
                    _expected_embedding_provenance=embedding_provenance,
                )
            except RunValidationError as exc:
                raise CampaignExecutionError(
                    f"new completion failed validation [{exc.code}] {directory}: {exc}"
                ) from exc
            if validated.key != key:
                raise CampaignExecutionError(
                    f"runner produced {validated.key.label()} while {key.label()} was requested"
                )
            if key in completed:
                raise CampaignExecutionError(f"duplicate completed key: {key.label()}")
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
                f"[campaign] validated {position}/{len(ordered_keys)} {key.label()}",
                flush=True,
            )

        if set(completed) != set(spec.expected_keys()):
            missing = sorted(set(spec.expected_keys()) - set(completed))
            raise CampaignExecutionError(
                "campaign loop ended with missing keys: "
                + ", ".join(key.label() for key in missing)
            )
        assert_source_snapshot_unchanged(workspace_root, source_snapshot)
        aggregate_command = [
            str(python_executable),
            str(workspace_root / "code" / "aggregate_corrected_campaign.py"),
            "--results-root",
            str(results_root),
            "--workspace-root",
            str(workspace_root),
        ]
        aggregate_log = log_root / (
            time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-aggregate.log"
        )
        with aggregate_log.open("x", encoding="utf-8", newline="\n") as log_handle:
            aggregate_return_code = _run_and_tee(
                aggregate_command, cwd=workspace_root, log_handle=log_handle
            )
        if aggregate_return_code != 0:
            raise CampaignExecutionError(
                f"independent campaign aggregation failed with exit code {aggregate_return_code}"
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
        help="Python interpreter used for every run and the final aggregation",
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
            python_executable=args.python.resolve(),
            device=args.device,
            dry_run=bool(args.dry_run),
        )
    except (CampaignExecutionError, OSError, ValueError) as exc:
        print(f"[campaign] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
