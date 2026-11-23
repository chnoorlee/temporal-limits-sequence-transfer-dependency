#!/usr/bin/env python3
"""Run the frozen 45-job family-transfer campaign on the designated server.

The launcher records every attempt and only resumes jobs whose complete output
passes the independent aggregator. It never removes failed or partial runs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import aggregate_family_inductive_campaign as aggregate


ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results" / "family_inductive_v2"
SOURCE_FILES = (
    "code/run_family_inductive_campaign.py",
    "code/run_family_inductive_experiment.py",
    "code/aggregate_family_inductive_campaign.py",
    "code/family_inductive.py",
    "code/run_family_holdout_experiment.py",
    "code/run_corrected_experiments.py",
    "code/corrected_protocol.py",
    "code/embedding_provenance.py",
    "code/family_holdout.py",
    "code/models.py",
    "docs/revision_v4/family_transfer_lock.json",
    "docs/revision_v4/FAMILY_TRANSFER_PROTOCOL.md",
)
SNAPSHOT_SCHEMA = "DepPLM-family-transfer-campaign-source-snapshot-v2"
PYTHON_ENVIRONMENT_SCHEMA = "DepPLM-family-transfer-python-environment-v1"
MIN_FREE_MIB = 8192
MAX_IDLE_UTILIZATION_PERCENT = 5
GPU_UUID = re.compile(r"GPU-[A-Za-z0-9][A-Za-z0-9-]{0,127}")
GPU_LOCK_NAMESPACE_PREFIX = "depplm-family-inductive-gpu-locks-uid"
PYTHON_ENVIRONMENT_RUNTIME_KEYS = {
    "numpy",
    "python",
    "scipy",
    "sklearn",
    "torch",
    "torch_cuda",
}
PYTHON_ENVIRONMENT_PROBE = """\
import json
import sys

import numpy
import scipy
import sklearn
import torch

payload = {
    "numpy": numpy.__version__,
    "python": sys.version,
    "scipy": scipy.__version__,
    "sklearn": sklearn.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}
print(json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False))
"""


class CampaignExecutionError(RuntimeError):
    """The campaign cannot continue without changing its frozen evidence."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"at_utc": _now(), "event": event, **fields}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _source_records(workspace_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        path = workspace_root / relative
        if path.is_symlink() or not path.is_file():
            raise CampaignExecutionError(f"missing or symlinked campaign source: {relative}")
        records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": aggregate._sha256_file(path),
        }
    return records


def _source_snapshot(
    *, campaign_id: str, roster_sha256: str, gpu_uuid: str, workspace_root: Path,
    python_environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SNAPSHOT_SCHEMA,
        "campaign_id": campaign_id,
        "roster_sha256": roster_sha256,
        "gpu_uuid": gpu_uuid,
        "python_environment": dict(python_environment),
        "source_files": _source_records(workspace_root),
        "external_outcomes_read": False,
    }


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _python_environment_record(python_executable: Path) -> dict[str, Any]:
    requested = Path(os.path.normpath(str(python_executable)))
    if not requested.is_absolute():
        raise CampaignExecutionError("--python must be an absolute path")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise CampaignExecutionError(
            f"--python cannot be resolved: {requested}"
        ) from exc
    if not resolved.is_file():
        raise CampaignExecutionError(f"--python does not resolve to a file: {resolved}")
    executable_sha256 = aggregate._sha256_file(resolved)
    try:
        result = subprocess.run(
            [str(requested), "-I", "-c", PYTHON_ENVIRONMENT_PROBE],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CampaignExecutionError("Python environment probe could not run") from exc
    if result.returncode != 0:
        raise CampaignExecutionError(
            "Python environment probe failed with exit code "
            f"{result.returncode}: {result.stderr.strip()}"
        )
    try:
        runtime = json.loads(
            result.stdout,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise CampaignExecutionError(
            "Python environment probe did not emit strict JSON"
        ) from exc
    if not isinstance(runtime, dict) or set(runtime) != PYTHON_ENVIRONMENT_RUNTIME_KEYS:
        raise CampaignExecutionError("Python environment probe schema mismatch")
    for field in PYTHON_ENVIRONMENT_RUNTIME_KEYS.difference({"torch_cuda"}):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise CampaignExecutionError(
                f"Python environment probe returned invalid {field}"
            )
    if runtime["torch_cuda"] is not None and (
        not isinstance(runtime["torch_cuda"], str) or not runtime["torch_cuda"]
    ):
        raise CampaignExecutionError(
            "Python environment probe returned invalid torch_cuda"
        )
    try:
        resolved_after = requested.resolve(strict=True)
    except OSError as exc:
        raise CampaignExecutionError("--python changed during environment probe") from exc
    if (
        resolved_after != resolved
        or aggregate._sha256_file(resolved_after) != executable_sha256
    ):
        raise CampaignExecutionError("--python changed during environment probe")
    return {
        "schema": PYTHON_ENVIRONMENT_SCHEMA,
        "requested_executable": str(requested),
        "resolved_executable": str(resolved),
        "resolved_executable_sha256": executable_sha256,
        "runtime": runtime,
    }


def _default_gpu_lock_root() -> Path:
    try:
        effective_uid = os.geteuid()
    except AttributeError as exc:
        raise CampaignExecutionError(
            "host-global GPU locking requires a Unix effective UID"
        ) from exc
    return Path("/tmp") / f"{GPU_LOCK_NAMESPACE_PREFIX}-{effective_uid}"


def _validate_gpu_lock_directory(metadata: os.stat_result, effective_uid: int) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise CampaignExecutionError("GPU campaign lock namespace is not a directory")
    if metadata.st_uid != effective_uid:
        raise CampaignExecutionError("GPU campaign lock namespace has an unsafe owner")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise CampaignExecutionError("GPU campaign lock namespace mode must be 0700")


def _validate_gpu_lock_file(metadata: os.stat_result, effective_uid: int) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise CampaignExecutionError("GPU campaign lock is not a regular file")
    if metadata.st_uid != effective_uid:
        raise CampaignExecutionError("GPU campaign lock has an unsafe owner")


def _acquire_gpu_campaign_lock(
    gpu_uuid: str, *, lock_namespace_root: Path | None = None,
) -> tuple[BinaryIO, Path]:
    """Lock one GPU across cooperating DepPLM launchers.

    This lock does not reserve the device against external jobs that do not
    participate in this protocol; the separate idle checks remain necessary.
    """
    if GPU_UUID.fullmatch(gpu_uuid) is None:
        raise CampaignExecutionError("unsafe GPU UUID for campaign lock")
    try:
        effective_uid = os.geteuid()
    except AttributeError as exc:
        raise CampaignExecutionError(
            "host-global GPU locking requires a Unix effective UID"
        ) from exc
    lock_root = (
        _default_gpu_lock_root()
        if lock_namespace_root is None
        else Path(lock_namespace_root)
    )
    if not lock_root.is_absolute():
        raise CampaignExecutionError("GPU campaign lock namespace must be absolute")
    try:
        lock_root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise CampaignExecutionError(
            "could not create GPU campaign lock namespace"
        ) from exc
    try:
        namespace_metadata = lock_root.lstat()
    except OSError as exc:
        raise CampaignExecutionError(
            "could not inspect GPU campaign lock namespace"
        ) from exc
    _validate_gpu_lock_directory(namespace_metadata, effective_uid)

    lock_name = f"{gpu_uuid}.lock"
    lock_path = lock_root / lock_name
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd = -1
    lock_fd = -1
    descriptor: BinaryIO | None = None
    try:
        directory_fd = os.open(lock_root, directory_flags)
        descriptor_metadata = os.fstat(directory_fd)
        _validate_gpu_lock_directory(descriptor_metadata, effective_uid)
        if (
            descriptor_metadata.st_dev != namespace_metadata.st_dev
            or descriptor_metadata.st_ino != namespace_metadata.st_ino
        ):
            raise CampaignExecutionError(
                "GPU campaign lock namespace changed while opening"
            )
        lock_fd = os.open(lock_name, file_flags, 0o600, dir_fd=directory_fd)
        lock_metadata = os.fstat(lock_fd)
        _validate_gpu_lock_file(lock_metadata, effective_uid)
        path_metadata = os.stat(
            lock_name, dir_fd=directory_fd, follow_symlinks=False
        )
        _validate_gpu_lock_file(path_metadata, effective_uid)
        if (
            path_metadata.st_dev != lock_metadata.st_dev
            or path_metadata.st_ino != lock_metadata.st_ino
        ):
            raise CampaignExecutionError("GPU campaign lock changed while opening")
        descriptor = os.fdopen(lock_fd, "a+b")
        lock_fd = -1
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            descriptor.close()
            descriptor = None
            raise CampaignExecutionError(
                f"another DepPLM launcher holds the {gpu_uuid} campaign lock"
            ) from exc
        except OSError as exc:
            descriptor.close()
            descriptor = None
            raise CampaignExecutionError(
                "could not acquire safe GPU campaign flock"
            ) from exc
        except BaseException:
            descriptor.close()
            descriptor = None
            raise
        return descriptor, lock_path
    except CampaignExecutionError:
        if descriptor is not None:
            descriptor.close()
        raise
    except OSError as exc:
        if descriptor is not None:
            descriptor.close()
        raise CampaignExecutionError("could not open safe GPU campaign lock") from exc
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        if directory_fd >= 0:
            os.close(directory_fd)


def _release_gpu_campaign_lock(descriptor: BinaryIO) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        descriptor.close()


def _load_or_create_snapshot(path: Path, expected: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise CampaignExecutionError("campaign source snapshot may not be a symlink")
    if not path.exists():
        aggregate.write_new_json(path, expected)
    stored = aggregate._load_json(path, label="campaign source snapshot")
    if stored != dict(expected):
        raise CampaignExecutionError(
            "campaign source snapshot differs from frozen source, Python environment, roster, or GPU"
        )


def _gpu_rows(output: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise CampaignExecutionError(f"unexpected nvidia-smi GPU row: {line!r}")
        try:
            index, free_mib, utilization = int(fields[0]), int(fields[2]), int(fields[3])
        except ValueError as exc:
            raise CampaignExecutionError(f"non-numeric nvidia-smi GPU row: {line!r}") from exc
        uuid = fields[1]
        if index in rows or not uuid.startswith("GPU-") or free_mib < 0 or utilization < 0:
            raise CampaignExecutionError(f"invalid nvidia-smi GPU row: {line!r}")
        rows[index] = {"index": index, "uuid": uuid, "free_mib": free_mib, "utilization_percent": utilization}
    if not rows:
        raise CampaignExecutionError("nvidia-smi returned no GPUs")
    return rows


def _compute_processes(output: str) -> dict[str, list[int]]:
    processes: dict[str, list[int]] = {}
    for line in output.splitlines():
        if line.strip() == "No running processes found":
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0].startswith("GPU-"):
            raise CampaignExecutionError(f"unexpected nvidia-smi process row: {line!r}")
        try:
            pid = int(fields[1])
        except ValueError as exc:
            raise CampaignExecutionError(f"invalid nvidia-smi PID: {line!r}") from exc
        if pid <= 0:
            raise CampaignExecutionError(f"invalid nvidia-smi PID: {line!r}")
        processes.setdefault(fields[0], []).append(pid)
    return processes


def probe_gpu(index: int) -> dict[str, Any]:
    commands = (
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
    )
    outputs = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise CampaignExecutionError(f"nvidia-smi failed with exit code {result.returncode}: {result.stderr.strip()}")
        outputs.append(result.stdout)
    rows = _gpu_rows(outputs[0])
    if index not in rows:
        raise CampaignExecutionError(f"requested GPU index {index} is unavailable")
    snapshot = rows[index]
    snapshot["compute_pids"] = sorted(_compute_processes(outputs[1]).get(snapshot["uuid"], []))
    return snapshot


def gpu_is_idle(snapshot: Mapping[str, Any]) -> bool:
    return (
        snapshot["free_mib"] >= MIN_FREE_MIB
        and snapshot["utilization_percent"] <= MAX_IDLE_UTILIZATION_PERCENT
        and not snapshot["compute_pids"]
    )


def wait_for_idle_gpu(
    *, index: int, uuid: str, ledger: Path, poll_seconds: int
) -> dict[str, Any]:
    last_wait_report = 0.0
    while True:
        snapshot = probe_gpu(index)
        if snapshot["uuid"] != uuid:
            raise CampaignExecutionError("selected GPU UUID changed during the frozen campaign")
        if gpu_is_idle(snapshot):
            time.sleep(10)
            confirmed = probe_gpu(index)
            if confirmed["uuid"] != uuid:
                raise CampaignExecutionError("selected GPU UUID changed during idle confirmation")
            if gpu_is_idle(confirmed):
                _append_event(ledger, "gpu_idle_confirmed", gpu=confirmed)
                return confirmed
        if time.monotonic() - last_wait_report >= 3600 or last_wait_report == 0:
            _append_event(ledger, "gpu_waiting", gpu=snapshot)
            last_wait_report = time.monotonic()
        time.sleep(poll_seconds)


def _validated_existing(
    *, runs_root: Path, roster: Mapping[str, aggregate.JobKey], roster_sha256: str,
    lock: Mapping[str, Any], lock_record: Mapping[str, Any], workspace_root: Path,
) -> dict[str, aggregate.ValidatedRun]:
    if runs_root.is_symlink():
        raise CampaignExecutionError("formal run root may not be a symlink")
    runs_root.mkdir(parents=True, exist_ok=True)
    entries = list(runs_root.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_dir() or entry.name not in roster:
            raise CampaignExecutionError(f"unexpected or unsafe formal run entry: {entry}")
    found: dict[str, aggregate.ValidatedRun] = {}
    for entry in sorted(entries, key=lambda path: path.name):
        found[entry.name] = aggregate.validate_run_directory(
            entry,
            expected_run_id=entry.name,
            expected_key=roster[entry.name],
            roster_sha256=roster_sha256,
            lock=lock,
            lock_record=lock_record,
            workspace_root=workspace_root,
            verify_workspace_files=True,
        )
    if found:
        aggregate._require_run_consistency(
            list(found.values()), workspace_root=workspace_root, verify_workspace_files=True
        )
    return found


def _run_command(
    *, python_executable: Path, workspace_root: Path, run_id: str,
    key: aggregate.JobKey, roster_sha256: str, runs_root: Path,
) -> list[str]:
    return [
        str(python_executable), str(workspace_root / "code" / "run_family_inductive_experiment.py"),
        "--identity-percent", str(int(key.identity_percent)),
        "--outer-fold", str(key.outer_fold),
        "--split-seed", str(key.split_seed),
        "--training-seed", str(key.training_seed),
        "--campaign-roster-sha256", roster_sha256,
        "--device", "cuda:0", "--run-id", run_id,
        "--output-root", str(runs_root),
    ]


def _run_and_log(command: Sequence[str], *, workspace_root: Path, log_path: Path, gpu_uuid: str) -> int:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    with log_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"command": list(command), "gpu_uuid": gpu_uuid}, sort_keys=True) + "\n")
        handle.flush()
        result = subprocess.run(
            list(command), cwd=workspace_root, env=environment, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        )
        handle.flush()
        os.fsync(handle.fileno())
    return result.returncode


def run_campaign(
    *, campaign_id: str, gpu_index: int = 1, python_executable: Path = Path(sys.executable),
    workspace_root: Path = ROOT, results_root: Path = RESULTS_ROOT, poll_seconds: int = 60,
    prepare_only: bool = False, gpu_lock_root: Path | None = None,
) -> Path | None:
    if aggregate.IDENTIFIER.fullmatch(campaign_id) is None or poll_seconds < 1:
        raise CampaignExecutionError("invalid campaign id or GPU poll interval")
    python_executable = Path(os.path.normpath(str(python_executable)))
    workspace_root = workspace_root.resolve(strict=True)
    lock, lock_record = aggregate.load_analysis_lock()
    aggregate._assert_designated_server(lock)
    if workspace_root != ROOT.resolve(strict=True):
        raise CampaignExecutionError("formal campaign must run in the canonical server worktree")
    if results_root.is_symlink():
        raise CampaignExecutionError("results root may not be a symlink")
    results_root.mkdir(parents=True, exist_ok=True)
    state_root = results_root / "campaign-state" / campaign_id
    state_root.mkdir(parents=True, exist_ok=True)
    ledger = state_root / "execution.jsonl"
    lock_path = state_root / "execution.lock"
    with lock_path.open("a+b") as descriptor:
        gpu_lock_descriptor: BinaryIO | None = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignExecutionError("another launcher holds this campaign lock") from exc
        try:
            first_gpu = probe_gpu(gpu_index)
            gpu_lock_descriptor, gpu_lock_path = _acquire_gpu_campaign_lock(
                str(first_gpu["uuid"]), lock_namespace_root=gpu_lock_root
            )
            _append_event(
                ledger,
                "gpu_campaign_lock_acquired",
                gpu_uuid=first_gpu["uuid"],
                lock_path=str(gpu_lock_path),
                scope="cooperating_DepPLM_launchers_only",
            )
            roster_path = state_root / "roster.json"
            if not roster_path.exists():
                aggregate.write_new_json(roster_path, aggregate.build_roster(campaign_id))
            roster, roster_sha256, indexed = aggregate.load_roster(
                roster_path, campaign_id=campaign_id, lock=lock, lock_record=lock_record
            )
            if len(indexed) != 45:
                raise CampaignExecutionError("roster is not the locked 45-job grid")
            snapshot_path = state_root / "source-snapshot.json"
            expected_snapshot = _source_snapshot(
                campaign_id=campaign_id, roster_sha256=roster_sha256,
                gpu_uuid=first_gpu["uuid"], workspace_root=workspace_root,
                python_environment=_python_environment_record(python_executable),
            )
            _load_or_create_snapshot(snapshot_path, expected_snapshot)
            runs_root = results_root / "formal-runs" / campaign_id
            completed = _validated_existing(
                runs_root=runs_root, roster=indexed, roster_sha256=roster_sha256,
                lock=lock, lock_record=lock_record, workspace_root=workspace_root,
            )
            _append_event(ledger, "campaign_checked", campaign_id=campaign_id,
                          roster_sha256=roster_sha256, completed=len(completed), pending=45-len(completed),
                          prepare_only=prepare_only)
            if prepare_only:
                return None
            log_root = state_root / "logs"
            for job in roster["jobs"]:
                run_id = job["run_id"]
                if run_id in completed:
                    continue
                _load_or_create_snapshot(snapshot_path, _source_snapshot(
                    campaign_id=campaign_id, roster_sha256=roster_sha256,
                    gpu_uuid=first_gpu["uuid"], workspace_root=workspace_root,
                    python_environment=_python_environment_record(python_executable),
                ))
                gpu = wait_for_idle_gpu(
                    index=gpu_index, uuid=first_gpu["uuid"], ledger=ledger, poll_seconds=poll_seconds
                )
                _load_or_create_snapshot(snapshot_path, _source_snapshot(
                    campaign_id=campaign_id, roster_sha256=roster_sha256,
                    gpu_uuid=first_gpu["uuid"], workspace_root=workspace_root,
                    python_environment=_python_environment_record(python_executable),
                ))
                log_root.mkdir(exist_ok=True)
                if (runs_root / run_id).exists() or (runs_root / run_id).is_symlink():
                    raise CampaignExecutionError(f"unvalidated run directory appeared: {run_id}")
                command = _run_command(
                    python_executable=python_executable, workspace_root=workspace_root,
                    run_id=run_id, key=indexed[run_id], roster_sha256=roster_sha256,
                    runs_root=runs_root,
                )
                log_path = log_root / f"{run_id}.log"
                _append_event(ledger, "run_started", run_id=run_id, run_key=job["run_key"],
                              gpu=gpu, log=str(log_path.relative_to(workspace_root)))
                started = time.monotonic()
                return_code = _run_and_log(
                    command, workspace_root=workspace_root, log_path=log_path,
                    gpu_uuid=first_gpu["uuid"],
                )
                if return_code != 0:
                    _append_event(ledger, "run_failed", run_id=run_id,
                                  return_code=return_code, wall_seconds=time.monotonic()-started)
                    raise CampaignExecutionError(f"{run_id} failed with exit code {return_code}")
                _load_or_create_snapshot(snapshot_path, _source_snapshot(
                    campaign_id=campaign_id, roster_sha256=roster_sha256,
                    gpu_uuid=first_gpu["uuid"], workspace_root=workspace_root,
                    python_environment=_python_environment_record(python_executable),
                ))
                completed[run_id] = aggregate.validate_run_directory(
                    runs_root / run_id, expected_run_id=run_id,
                    expected_key=indexed[run_id], roster_sha256=roster_sha256,
                    lock=lock, lock_record=lock_record, workspace_root=workspace_root,
                    verify_workspace_files=True,
                )
                _append_event(ledger, "run_validated", run_id=run_id,
                              completion_sha256=aggregate._sha256_file(runs_root / run_id / "completion.json"),
                              completed=len(completed), wall_seconds=time.monotonic()-started)
            if set(completed) != set(indexed):
                raise CampaignExecutionError("campaign ended without all 45 validated jobs")
            _load_or_create_snapshot(snapshot_path, _source_snapshot(
                campaign_id=campaign_id, roster_sha256=roster_sha256,
                gpu_uuid=first_gpu["uuid"], workspace_root=workspace_root,
                python_environment=_python_environment_record(python_executable),
            ))
            metrics, consensus, input_manifest = aggregate.validate_campaign(
                campaign_id=campaign_id, roster_path=roster_path,
                runs_root=runs_root, workspace_root=workspace_root,
            )
            destination = aggregate.publish_campaign(
                campaign_id=campaign_id, metrics=metrics, consensus=consensus,
                input_manifest=input_manifest, output_root=results_root / "campaigns",
            )
            _append_event(ledger, "campaign_aggregated", destination=str(destination),
                          completed=len(completed),
                          completion_sha256=aggregate._sha256_file(destination / "completion.json"))
            return destination
        except BaseException as exc:
            _append_event(ledger, "campaign_failed", error_type=type(exc).__name__,
                          message=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            if gpu_lock_descriptor is not None:
                _release_gpu_campaign_lock(gpu_lock_descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_UN)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        destination = run_campaign(
            campaign_id=args.campaign_id, gpu_index=args.gpu_index,
            python_executable=args.python, poll_seconds=args.poll_seconds,
            prepare_only=args.prepare_only,
        )
        print("[family-transfer-campaign] prepared" if destination is None else destination, flush=True)
        return 0
    except (OSError, ValueError, CampaignExecutionError,
            aggregate.CampaignValidationError, aggregate.AtomicPublishError) as exc:
        print(f"[family-transfer-campaign] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
