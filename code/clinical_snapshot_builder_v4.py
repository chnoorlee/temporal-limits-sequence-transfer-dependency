"""Hash-bound Open Targets 26.09 clinical-snapshot builder and verifier.

This module owns the complete raw-input validation, endpoint derivation,
serialization, inventory, and rederivation contract.  Clinical analysis code may
evolve independently; a v4 snapshot remains bound to these exact builder bytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import stat
import sys
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

_SRC_SHA256 = "7f4233c4a1f9b417293e09d0657f1e010000eaf309bb622415bbc9a95c83d80c"

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_ROOT = ROOT / "data" / "external" / "opentargets_26_09" / "raw"
DEFAULT_OUTPUT_DIR = (
    ROOT / "data" / "external" / "opentargets_26_09" / "frozen_v4_final"
)
RELEASE = "26.09"
SNAPSHOT_SCHEMA = "DepPLM-OpenTargets-26.09-clinical-snapshot-v4"
LEGACY_SNAPSHOT_SCHEMA = "DepPLM-OpenTargets-26.09-clinical-snapshot-v3"
BUILDER_API = "clinical-snapshot-builder-v4.0"
SNAPSHOT_RESULT_REASON = (
    "External-data snapshot only; no DepPLM prediction was analysed."
)
CANCER_ROOT = "MONDO_0045024"
PHASE2_PLUS_STAGES = (
    "PHASE_2",
    "PHASE_2_3",
    "PHASE_3",
    "PHASE_4",
    "PREAPPROVAL",
    "APPROVAL",
)
WITHDRAWAL_STAGE = "WITHDRAWAL"
STAGE_ORDER = {stage: index for index, stage in enumerate(PHASE2_PLUS_STAGES, 1)}
OUTCOME_FIELDS = (
    "approved_symbol",
    "target_id",
    "clinical_oncology_phase2_plus",
    "oncology_phase2_plus_drug_count",
    "oncology_phase2_plus_stages",
    "maximum_oncology_phase2_plus_stage",
    "clinical_oncology_withdrawal",
    "oncology_withdrawn_drug_count",
    "tractable_any",
    "tractability_true_record_count",
    "tractability_true_modalities",
)
DISEASE_FIELDS = ("disease_id", "name", "scope_relation")
AMBIGUOUS_FIELDS = ("approved_symbol", "target_ids", "reason")
ARTIFACT_FIELDS = {
    "target_outcomes.tsv": OUTCOME_FIELDS,
    "disease_scope.tsv": DISEASE_FIELDS,
    "ambiguous_symbols.tsv": AMBIGUOUS_FIELDS,
}
LEGACY_V3_GENERATOR = {
    "bytes": 84_452,
    "sha256": "e1873ec36eb7f9ce1f7bb6f462f36f8e886feb91f0f2e19281add34ecae4a68f",
    "python": "3.13.9",
    "pyarrow": "21.0.0",
}


class SnapshotValidationError(RuntimeError):
    """Raised when the snapshot build or validation contract fails."""


def _source_binding_sha256(payload: bytes) -> str:
    """Hash exact source bytes after blanking only their embedded digest value."""
    marker = b"_SRC_" + b'SHA256 = "'
    start = payload.find(marker)
    if start < 0 or payload.find(marker, start + 1) >= 0:
        raise SnapshotValidationError(
            "snapshot builder source binding marker is noncanonical"
        )
    value_start = start + len(marker)
    value_end = value_start + 64
    value = payload[value_start:value_end]
    if (
        len(value) != 64
        or payload[value_end : value_end + 1] != b'"'
        or any(byte not in b"0123456789abcdef" for byte in value)
    ):
        raise SnapshotValidationError(
            "snapshot builder source binding value is noncanonical"
        )
    if value.decode("ascii") != _SRC_SHA256:
        raise SnapshotValidationError(
            "snapshot builder source binding value differs from the executing module"
        )
    normalized = payload[:value_start] + (b"0" * 64) + payload[value_end:]
    return hashlib.sha256(normalized).hexdigest()


def _capture_module_source(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Bind bytes cryptographically identified by the executing module payload."""
    lexical = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lexical, flags)
    try:
        opened = os.fstat(descriptor)
        named = os.stat(lexical, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(opened, named):
            raise SnapshotValidationError(
                "snapshot builder source is not a stable regular file"
            )
        digest = hashlib.sha256()
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            payload.extend(chunk)
        after = os.fstat(descriptor)
        named_after = os.stat(lexical, follow_symlinks=False)
        if (
            not os.path.samestat(opened, after)
            or not os.path.samestat(opened, named_after)
            or after.st_size != len(payload)
        ):
            raise SnapshotValidationError(
                "snapshot builder source changed while being bound"
            )
    finally:
        os.close(descriptor)
    bound_payload = bytes(payload)
    if _source_binding_sha256(bound_payload) != _SRC_SHA256:
        raise SnapshotValidationError(
            "snapshot builder source payload does not match the executing module binding"
        )
    return bound_payload, {
        "bytes": len(payload),
        "sha256": digest.hexdigest(),
    }


_LOADED_SOURCE_PATH = Path(__file__).resolve()
_LOADED_SOURCE_PAYLOAD, _LOADED_SOURCE_FINGERPRINT = _capture_module_source(
    _LOADED_SOURCE_PATH
)


def _require_loaded_source_unchanged() -> None:
    """Reject a stale loaded module if its on-disk source name now binds other bytes."""
    try:
        _payload, observed = _capture_module_source(_LOADED_SOURCE_PATH)
    except SnapshotValidationError as exc:
        raise SnapshotValidationError(
            "snapshot builder source changed after this module was loaded"
        ) from exc
    if observed != _LOADED_SOURCE_FINGERPRINT:
        raise SnapshotValidationError(
            "snapshot builder source changed after this module was loaded"
        )


@dataclass(frozen=True)
class RawFileSpec:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class CapturedFile:
    """One exact-byte, builder-owned capture of a source file."""

    source_path: Path
    path: Path
    bytes: int
    sha256: str
    identity: tuple[int, int]


class _OwnedWindowsHandle:
    """Own a Win32 HANDLE from before a native out-parameter is populated."""

    __slots__ = ("_slot",)

    def __init__(self, raw_handle: int | None = None) -> None:
        from ctypes import wintypes

        self._slot = wintypes.HANDLE(raw_handle)

    @property
    def value(self) -> int:
        value = self._slot.value
        if value is None:
            raise RuntimeError("Windows handle is not owned")
        return int(value)

    @property
    def pointer(self) -> Any:
        import ctypes

        return ctypes.byref(self._slot)

    def release(self) -> int:
        value = self.value
        self._slot.value = None
        return value

    def close(self) -> int | None:
        value = self._slot.value
        if value is None:
            return None
        self._slot.value = None
        return _windows_close_handle(int(value))

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


@dataclass
class _PinnedDirectory:
    """An opened directory identity held across publication or cleanup."""

    path: Path
    identity: tuple[int, int]
    windows_handle: _OwnedWindowsHandle | None = None
    posix_fd: int | None = None


@dataclass(frozen=True)
class _PendingSnapshotPublication:
    """An exclusively reserved snapshot directory awaiting its commit marker."""

    destination: Path
    staging: Path
    parent_identity: tuple[int, int]
    staging_identity: tuple[int, int]
    parent_pin: _PinnedDirectory
    staging_pin: _PinnedDirectory
    created_file_identities: dict[str, tuple[int, int]] = field(default_factory=dict)


RAW_FILES = (
    RawFileSpec(
        "clinical_target/00000000.parquet",
        2_815_006,
        "33feae900b97a0935c3e30a5c745d3074a2ca46c5e50456b334482c0da113735",
    ),
    RawFileSpec(
        "disease/00000000.parquet",
        6_346_458,
        "a89b8b8b89bad168016e6ff8302d2bcc8d9b7f9cbb35c459ef847c36a4f17aed",
    ),
    RawFileSpec(
        "target/part-00000-ddfc1f8c-60a1-40fe-885a-2c1dd3f9a9c1-c000.zstd.parquet",
        7_165_956,
        "80186f1c9919d5d1351f009253480fad93b3bb3e2de506d8948ed2126ac48cab",
    ),
    RawFileSpec(
        "target/part-00001-ddfc1f8c-60a1-40fe-885a-2c1dd3f9a9c1-c000.zstd.parquet",
        7_196_171,
        "a777ec38d5fbf6e6475d85158141c0a01f2c32cadc7f8769256eaab6e034e702",
    ),
    RawFileSpec(
        "target/part-00002-ddfc1f8c-60a1-40fe-885a-2c1dd3f9a9c1-c000.zstd.parquet",
        7_270_990,
        "3621b2510eb3f9da9b533f40559b4f4de390fffc57935726a2b0eebcc40aa028",
    ),
    RawFileSpec(
        "target_tractability/part-00000-e2201205-916c-4b28-971d-6511ae9ea691-c000.zstd.parquet",
        55_708,
        "c81b9d1c61d3fa1e23e67a5b91c082e7d20829aba578ffe95adc9ea1ef2332fe",
    ),
    RawFileSpec(
        "target_tractability/part-00001-e2201205-916c-4b28-971d-6511ae9ea691-c000.zstd.parquet",
        72_189,
        "ebc49a49ae182dee29be43664dfc61c21897bf3d776d9df18f7e0e73bc0990ea",
    ),
)


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_bytes(payload: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotValidationError(
                    f"{label} contains duplicate JSON key: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise SnapshotValidationError(
            f"{label} contains non-finite JSON constant: {value}"
        )

    try:
        result = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except SnapshotValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotValidationError(f"{label} is not valid UTF-8 JSON") from exc

    def reject_nonfinite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise SnapshotValidationError(f"{label} contains a non-finite number")
        if isinstance(value, dict):
            for nested in value.values():
                reject_nonfinite(nested)
        elif isinstance(value, list):
            for nested in value:
                reject_nonfinite(nested)

    reject_nonfinite(result)
    return result


def _require_canonical_utc_timestamp(value: Any, *, label: str) -> None:
    if not isinstance(value, str):
        raise SnapshotValidationError(f"{label} is not a canonical UTC timestamp")
    try:
        parsed = time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise SnapshotValidationError(
            f"{label} is not a canonical UTC timestamp"
        ) from exc
    if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) != value:
        raise SnapshotValidationError(f"{label} is not a canonical UTC timestamp")


def _record_for_bytes(payload: bytes, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    primary_error: BaseException | None = None
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error: OSError | None = None
        for _attempt in range(_CLEANUP_ATTEMPTS):
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_error = exc
                if _attempt + 1 < _CLEANUP_ATTEMPTS:
                    time.sleep(0.01 * (_attempt + 1))
                continue
            cleanup_error = None
            break
        if cleanup_error is not None:
            message = f"could not remove atomic-write temporary file: {temporary}"
            if primary_error is None:
                raise SnapshotValidationError(message) from cleanup_error
            primary_error.add_note(f"atomic-write cleanup also failed: {message}")


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    label = (
        resolved.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": label,
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _verify_file_record(path: Path, record: Mapping[str, Any], *, label: str) -> None:
    if set(record) != {"path", "bytes", "sha256"}:
        raise SnapshotValidationError(f"{label} record is noncanonical")
    expected_bytes = record.get("bytes")
    expected_sha = record.get("sha256")
    if (
        not path.is_file()
        or path.is_symlink()
        or bool(
            getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise SnapshotValidationError(f"{label} binding is invalid: {path}")
    if path.stat().st_size != expected_bytes:
        raise SnapshotValidationError(f"{label} byte count changed: {path}")
    if _sha256_file(path) != expected_sha:
        raise SnapshotValidationError(f"{label} SHA-256 changed: {path}")


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_link_or_reparse_components(path: Path, *, label: str) -> None:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SnapshotValidationError(
                f"cannot inspect {label} path component: {current}"
            ) from exc
        if current.is_symlink() or bool(
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise SnapshotValidationError(
                f"{label} contains a symlink or reparse point: {current}"
            )


def _open_regular_source_once(path: Path, *, label: str) -> Any:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SnapshotValidationError(f"cannot open {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        lexical = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not os.path.samestat(opened, lexical)
            or bool(
                getattr(lexical, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
        ):
            raise SnapshotValidationError(
                f"{label} is not a stable regular file: {path}"
            )
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _capture_file_to_private(
    source: Path,
    destination: Path,
    *,
    label: str,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    chunk_size: int = 4 * 1024 * 1024,
    _payload_sink: bytearray | None = None,
    _destination_parent_pin: _PinnedDirectory,
) -> CapturedFile:
    """Capture ``source`` while the caller retains the private parent pin."""
    source = _absolute_without_resolving(source)
    _reject_link_or_reparse_components(source, label=label)
    digest = hashlib.sha256()
    byte_count = 0
    destination_parent = _destination_parent_pin
    created_identity: tuple[int, int] | None = None
    try:
        with _open_regular_source_once(source, label=label) as source_handle:
            with _open_pinned_exclusive_file(
                destination_parent,
                destination.name,
                label="private raw-data capture file",
            ) as destination_handle:
                created_metadata = os.fstat(destination_handle.fileno())
                created_identity = (
                    int(created_metadata.st_dev),
                    int(created_metadata.st_ino),
                )
                while chunk := source_handle.read(chunk_size):
                    destination_handle.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
                    if _payload_sink is not None:
                        _payload_sink.extend(chunk)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        observed_sha256 = digest.hexdigest()
        if expected_bytes is not None and byte_count != expected_bytes:
            raise SnapshotValidationError(f"{label} byte count changed: {source}")
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise SnapshotValidationError(f"{label} SHA-256 changed: {source}")
        if created_identity is None:
            raise SnapshotValidationError(
                "private raw-data capture identity was not recorded"
            )
        return CapturedFile(
            source_path=source,
            path=destination,
            bytes=byte_count,
            sha256=observed_sha256,
            identity=created_identity,
        )
    except BaseException as exc:
        try:
            if created_identity is not None:
                _delete_pinned_file(
                    destination_parent,
                    destination.name,
                    created_identity,
                    label="partial or rejected private raw-data capture",
                )
        except Exception as cleanup_error:
            exc.add_note(
                "private raw-data capture cleanup also failed: " f"{cleanup_error}"
            )
        raise


_CLEANUP_ATTEMPTS = 3


def _directory_identity(path: Path, *, label: str) -> tuple[int, int]:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SnapshotValidationError(f"cannot inspect {label}: {path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or bool(
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    ):
        raise SnapshotValidationError(f"{label} is not a stable directory: {path}")
    return int(metadata.st_dev), int(metadata.st_ino)


def _require_directory_identity(
    path: Path, expected: tuple[int, int], *, label: str
) -> None:
    if _directory_identity(path, label=label) != expected:
        raise SnapshotValidationError(f"{label} identity changed: {path}")


def _pin_directory(path: Path, *, label: str) -> _PinnedDirectory:
    """Open a directory without following links and retain its identity."""
    expected = _directory_identity(path, label=label)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            str(path),
            0x00010000 | 0x00000080 | 0x00000001,
            # DELETE | FILE_READ_ATTRIBUTES | FILE_LIST_DIRECTORY
            0x00000001 | 0x00000002,  # FILE_SHARE_READ | FILE_SHARE_WRITE
            None,
            3,  # OPEN_EXISTING
            0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            error = ctypes.get_last_error()
            raise SnapshotValidationError(f"cannot pin {label}: {path}") from OSError(
                error, os.strerror(error), str(path)
            )
        handle_owner = _OwnedWindowsHandle(int(handle))
        pin = _PinnedDirectory(
            path=path, identity=expected, windows_handle=handle_owner
        )
        try:
            _require_directory_identity(path, expected, label=label)
        except BaseException:
            handle_owner.close()
            raise
        return pin

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SnapshotValidationError(f"cannot pin {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (int(opened.st_dev), int(opened.st_ino)) != expected
        ):
            raise SnapshotValidationError(
                f"{label} identity changed while opening: {path}"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return _PinnedDirectory(path=path, identity=expected, posix_fd=descriptor)


def _close_pinned_directory(pin: _PinnedDirectory, *, label: str) -> None:
    if pin.windows_handle is not None:
        handle = pin.windows_handle
        pin.windows_handle = None
        error = handle.close()
        if error is not None:
            raise SnapshotValidationError(
                f"could not close {label} handle"
            ) from OSError(error, os.strerror(error), str(pin.path))
    if pin.posix_fd is not None:
        descriptor = pin.posix_fd
        pin.posix_fd = None
        try:
            os.close(descriptor)
        except OSError as exc:
            raise SnapshotValidationError(
                f"could not close {label} descriptor"
            ) from exc


def _require_leaf_name(name: str, *, label: str) -> None:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise SnapshotValidationError(f"{label} name is noncanonical")


def _windows_open_relative_handle(
    directory: _PinnedDirectory,
    name: str,
    *,
    desired_access: int,
    share_access: int,
    create_disposition: int,
    create_options: int,
    label: str,
) -> _OwnedWindowsHandle:
    """Open a child relative to a retained Windows directory handle."""
    import ctypes
    from ctypes import wintypes

    _require_leaf_name(name, label=label)
    if directory.windows_handle is None:
        raise SnapshotValidationError(f"{label} directory handle is closed")

    class UnicodeString(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", wintypes.LPWSTR),
        ]

    class ObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.ULONG),
            ("RootDirectory", wintypes.HANDLE),
            ("ObjectName", ctypes.POINTER(UnicodeString)),
            ("Attributes", wintypes.ULONG),
            ("SecurityDescriptor", wintypes.LPVOID),
            ("SecurityQualityOfService", wintypes.LPVOID),
        ]

    class IoStatusBlock(ctypes.Structure):
        _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]

    name_buffer = ctypes.create_unicode_buffer(name)
    encoded_length = len(name.encode("utf-16-le"))
    object_name = UnicodeString(
        encoded_length,
        encoded_length + ctypes.sizeof(ctypes.c_wchar),
        ctypes.cast(name_buffer, wintypes.LPWSTR),
    )
    attributes = ObjectAttributes(
        ctypes.sizeof(ObjectAttributes),
        wintypes.HANDLE(directory.windows_handle.value),
        ctypes.pointer(object_name),
        0x00000040,
        None,
        None,
    )
    io_status = IoStatusBlock()
    handle_owner = _OwnedWindowsHandle()
    nt_create_file = ctypes.WinDLL("ntdll", use_last_error=True).NtCreateFile
    nt_create_file.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(ObjectAttributes),
        ctypes.POINTER(IoStatusBlock),
        ctypes.c_void_p,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        ctypes.c_void_p,
        wintypes.ULONG,
    ]
    nt_create_file.restype = ctypes.c_long
    status = nt_create_file(
        handle_owner.pointer,
        desired_access,
        ctypes.byref(attributes),
        ctypes.byref(io_status),
        None,
        0x00000080,
        share_access,
        create_disposition,
        create_options,
        None,
        0,
    )
    if status < 0:
        rtl_error = ctypes.WinDLL("ntdll").RtlNtStatusToDosError
        rtl_error.argtypes = [ctypes.c_long]
        rtl_error.restype = wintypes.ULONG
        error = int(rtl_error(status))
        raise OSError(error, os.strerror(error), str(directory.path / name))
    try:
        handle_owner.value
    except RuntimeError:
        raise SnapshotValidationError(f"could not open {label}: {name}")
    return handle_owner


def _windows_list_pinned_directory_names(
    directory: _PinnedDirectory, *, label: str
) -> list[str]:
    """Enumerate one retained Windows directory handle without reopening its path."""
    import ctypes
    from ctypes import wintypes

    if directory.windows_handle is None:
        raise SnapshotValidationError(f"{label} directory handle is closed")

    class IoStatusBlock(ctypes.Structure):
        _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]

    query = ctypes.WinDLL("ntdll", use_last_error=True).NtQueryDirectoryFile
    query.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(IoStatusBlock),
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.c_int,
        wintypes.BOOLEAN,
        ctypes.c_void_p,
        wintypes.BOOLEAN,
    ]
    query.restype = ctypes.c_long
    names: list[str] = []
    seen: set[str] = set()
    restart = True
    while True:
        buffer = ctypes.create_string_buffer(64 * 1024)
        io_status = IoStatusBlock()
        status = query(
            wintypes.HANDLE(directory.windows_handle.value),
            None,
            None,
            None,
            ctypes.byref(io_status),
            buffer,
            len(buffer),
            12,  # FileNamesInformation
            False,
            None,
            restart,
        )
        status_code = ctypes.c_ulong(status).value
        if status_code == 0x80000006:  # STATUS_NO_MORE_FILES
            break
        if status_code != 0:
            rtl_error = ctypes.WinDLL("ntdll").RtlNtStatusToDosError
            rtl_error.argtypes = [ctypes.c_long]
            rtl_error.restype = wintypes.ULONG
            error = int(rtl_error(status))
            raise SnapshotValidationError(
                f"could not enumerate {label}: {directory.path}"
            ) from OSError(error, os.strerror(error), str(directory.path))
        used = int(io_status.Information)
        if used <= 0 or used > len(buffer):
            raise SnapshotValidationError(f"{label} directory listing is invalid")
        offset = 0
        while True:
            if offset + 12 > used:
                raise SnapshotValidationError(f"{label} directory listing is invalid")
            next_offset = int.from_bytes(buffer.raw[offset : offset + 4], "little")
            name_bytes = int.from_bytes(buffer.raw[offset + 8 : offset + 12], "little")
            end = offset + 12 + name_bytes
            if name_bytes % 2 or end > used:
                raise SnapshotValidationError(f"{label} directory listing is invalid")
            try:
                name = buffer.raw[offset + 12 : end].decode("utf-16-le")
            except UnicodeDecodeError as exc:
                raise SnapshotValidationError(
                    f"{label} directory listing contains an invalid name"
                ) from exc
            if name not in {".", ".."}:
                _require_leaf_name(name, label=label)
                folded = name.casefold()
                if folded in seen:
                    raise SnapshotValidationError(
                        f"{label} directory listing contains a duplicate name"
                    )
                seen.add(folded)
                names.append(name)
            if next_offset == 0:
                break
            if next_offset < 12 or next_offset % 4 or offset + next_offset >= used:
                raise SnapshotValidationError(f"{label} directory listing is invalid")
            offset += next_offset
        restart = False
    return names


def _windows_relative_entry_metadata(
    directory: _PinnedDirectory, name: str, *, label: str
) -> os.stat_result:
    """Inspect a child through the retained parent handle without a path reopen."""
    try:
        handle_owner = _windows_open_relative_handle(
            directory,
            name,
            desired_access=0x00000080 | 0x00100000,
            share_access=0x00000001 | 0x00000002,
            create_disposition=1,
            create_options=0x00000020 | 0x00200000,
            label=label,
        )
    except OSError as exc:
        raise SnapshotValidationError(f"could not inspect {label}: {name}") from exc
    return _windows_fstat_owned_handle(handle_owner, label=label, display=name)


def _windows_close_handle(raw_handle: int) -> int | None:
    import ctypes
    from ctypes import wintypes

    close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    if not close_handle(wintypes.HANDLE(raw_handle)):
        return int(ctypes.get_last_error())
    return None


def _windows_fstat_owned_handle(
    handle: int | _OwnedWindowsHandle, *, label: str, display: str | Path
) -> os.stat_result:
    """Consume one Win32 handle and close it even if CRT adoption or fstat fails."""
    import msvcrt

    if isinstance(handle, _OwnedWindowsHandle):
        handle_owner: _OwnedWindowsHandle | None = handle
        raw_handle = handle.value
    else:
        handle_owner = None
        raw_handle = handle
    descriptor: int | None = None
    try:
        try:
            descriptor = msvcrt.open_osfhandle(raw_handle, os.O_RDONLY)
            if handle_owner is not None:
                handle_owner.release()
        except BaseException:
            if handle_owner is None:
                _windows_close_handle(raw_handle)
            else:
                handle_owner.close()
            raise
        return os.fstat(descriptor)
    except OSError as exc:
        raise SnapshotValidationError(f"could not inspect {label}: {display}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _windows_duplicate_handle_metadata(
    raw_handle: int, *, label: str, display: str | Path
) -> os.stat_result:
    """fstat a duplicate while leaving the caller's retained handle open."""
    import ctypes
    from ctypes import wintypes

    duplicate = wintypes.HANDLE()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.DuplicateHandle.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.DuplicateHandle.restype = wintypes.BOOL
    current_process = kernel32.GetCurrentProcess()
    if not kernel32.DuplicateHandle(
        current_process,
        wintypes.HANDLE(raw_handle),
        current_process,
        ctypes.byref(duplicate),
        0,
        False,
        0x00000002,
    ):
        error = ctypes.get_last_error()
        raise SnapshotValidationError(
            f"could not inspect {label}: {display}"
        ) from OSError(error, os.strerror(error), str(display))
    duplicate_value = duplicate.value
    if duplicate_value is None:
        raise SnapshotValidationError(f"could not inspect {label}: {display}")
    return _windows_fstat_owned_handle(
        int(duplicate_value), label=label, display=display
    )


def _create_pinned_directory(
    parent: _PinnedDirectory, name: str, path: Path, *, label: str
) -> _PinnedDirectory:
    """Create a child exclusively and retain the identity observed after opening.

    POSIX ``mkdirat`` does not return a directory descriptor.  The checks below
    therefore provide point-in-time name-to-handle binding, not immunity to a
    same-UID replacement between ``mkdirat`` and the first ``openat``.
    """
    _require_leaf_name(name, label=label)
    if os.name == "nt":
        try:
            handle_owner = _windows_open_relative_handle(
                parent,
                name,
                desired_access=(0x00010000 | 0x00000080 | 0x00100000 | 0x00000001),
                share_access=0x00000001 | 0x00000002,
                create_disposition=2,
                create_options=0x00000001 | 0x00000020 | 0x00200000,
                label=label,
            )
        except OSError as exc:
            if getattr(exc, "winerror", None) in (80, 183) or exc.errno in (
                17,
                80,
                183,
            ):
                raise FileExistsError(str(path)) from exc
            raise SnapshotValidationError(f"could not create {label}: {path}") from exc
        try:
            metadata = _windows_duplicate_handle_metadata(
                handle_owner.value, label=label, display=path
            )
        except BaseException:
            handle_owner.close()
            raise
        return _PinnedDirectory(
            path=path,
            identity=(int(metadata.st_dev), int(metadata.st_ino)),
            windows_handle=handle_owner,
        )

    if parent.posix_fd is None:
        raise SnapshotValidationError(f"{label} parent descriptor is closed")
    os.mkdir(name, 0o700, dir_fd=parent.posix_fd)
    created = os.stat(name, dir_fd=parent.posix_fd, follow_symlinks=False)
    if not stat.S_ISDIR(created.st_mode):
        raise SnapshotValidationError(f"{label} creation identity is invalid")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent.posix_fd)
    try:
        metadata = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent.posix_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or not os.path.samestat(created, metadata)
            or not os.path.samestat(metadata, current)
        ):
            raise SnapshotValidationError(f"{label} identity changed while opening")
        pin = _PinnedDirectory(
            path=path,
            identity=(int(metadata.st_dev), int(metadata.st_ino)),
            posix_fd=descriptor,
        )
        _require_pinned_directory_entry(
            parent, name, pin.identity, label=f"{label} final creation binding"
        )
        return pin
    except BaseException:
        os.close(descriptor)
        raise


def _open_pinned_child_directory(
    parent: _PinnedDirectory,
    name: str,
    path: Path,
    *,
    label: str,
    expected_identity: tuple[int, int] | None = None,
) -> _PinnedDirectory:
    """Open one existing child relative to ``parent`` and pin that identity."""
    _require_leaf_name(name, label=label)
    if os.name == "nt":
        try:
            handle_owner = _windows_open_relative_handle(
                parent,
                name,
                desired_access=(0x00010000 | 0x00000080 | 0x00100000 | 0x00000001),
                share_access=0x00000001 | 0x00000002,
                create_disposition=1,
                create_options=0x00000001 | 0x00000020 | 0x00200000,
                label=label,
            )
        except OSError as exc:
            raise SnapshotValidationError(f"could not open {label}: {path}") from exc
        try:
            metadata = _windows_duplicate_handle_metadata(
                handle_owner.value, label=label, display=path
            )
        except BaseException:
            handle_owner.close()
            raise
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if not stat.S_ISDIR(metadata.st_mode) or (
            expected_identity is not None and identity != expected_identity
        ):
            handle_owner.close()
            raise SnapshotValidationError(f"{label} identity changed: {path}")
        return _PinnedDirectory(
            path=path,
            identity=identity,
            windows_handle=handle_owner,
        )

    if parent.posix_fd is None:
        raise SnapshotValidationError(f"{label} parent descriptor is closed")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent.posix_fd)
    except OSError as exc:
        raise SnapshotValidationError(f"could not open {label}: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if not stat.S_ISDIR(metadata.st_mode) or (
            expected_identity is not None and identity != expected_identity
        ):
            raise SnapshotValidationError(f"{label} identity changed: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return _PinnedDirectory(path=path, identity=identity, posix_fd=descriptor)


def _require_pinned_directory_entry(
    parent: _PinnedDirectory,
    name: str,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    if os.name == "nt":
        try:
            handle_owner = _windows_open_relative_handle(
                parent,
                name,
                desired_access=0x00000080 | 0x00100000,
                share_access=0x00000001 | 0x00000002 | 0x00000004,
                create_disposition=1,
                create_options=0x00000001 | 0x00000020 | 0x00200000,
                label=label,
            )
        except OSError as exc:
            raise SnapshotValidationError(f"could not inspect {label}: {name}") from exc
        try:
            metadata = _windows_duplicate_handle_metadata(
                handle_owner.value, label=label, display=name
            )
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or (int(metadata.st_dev), int(metadata.st_ino)) != expected_identity
            ):
                raise SnapshotValidationError(f"{label} identity changed: {name}")
        finally:
            handle_owner.close()
        return
    child = _open_pinned_child_directory(
        parent,
        name,
        parent.path / name,
        label=label,
        expected_identity=expected_identity,
    )
    _close_pinned_directory(child, label=label)


def _open_pinned_exclusive_file(
    directory: _PinnedDirectory, name: str, *, label: str
) -> Any:
    """Create one file under an open directory without following a leaf link."""
    _require_leaf_name(name, label=label)
    if os.name == "nt":
        import msvcrt

        try:
            handle_owner = _windows_open_relative_handle(
                directory,
                name,
                desired_access=(0x40000000 | 0x00010000 | 0x00000080 | 0x00100000),
                share_access=0x00000001,
                create_disposition=2,
                create_options=0x00000040 | 0x00000020 | 0x00200000,
                label=label,
            )
        except OSError as exc:
            raise SnapshotValidationError(f"could not create {label}: {name}") from exc
        try:
            descriptor = msvcrt.open_osfhandle(
                handle_owner.value, os.O_WRONLY | getattr(os, "O_BINARY", 0)
            )
            handle_owner.release()
        except BaseException:
            handle_owner.close()
            raise
        try:
            return os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise

    if directory.posix_fd is None:
        raise SnapshotValidationError(f"{label} directory descriptor is closed")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory.posix_fd)
    except OSError as exc:
        raise SnapshotValidationError(f"could not create {label}: {name}") from exc
    try:
        return os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        raise


def _write_pinned_bytes(
    directory: _PinnedDirectory,
    name: str,
    payload: bytes,
    identities: dict[str, tuple[int, int]],
    *,
    label: str,
) -> None:
    """Write final bytes once and remember the exact created file identity."""
    with _open_pinned_exclusive_file(directory, name, label=label) as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise SnapshotValidationError(f"{label} is not a regular file: {name}")
        identities[name] = (int(metadata.st_dev), int(metadata.st_ino))
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _require_pinned_file_identity(
    directory: _PinnedDirectory,
    name: str,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    _require_leaf_name(name, label=label)
    descriptor: int | None = None
    handle_owner: _OwnedWindowsHandle | None = None
    try:
        if os.name == "nt":
            import msvcrt

            handle_owner = _windows_open_relative_handle(
                directory,
                name,
                desired_access=0x00000080 | 0x00100000,
                share_access=0x00000001 | 0x00000002,
                create_disposition=1,
                create_options=0x00000040 | 0x00000020 | 0x00200000,
                label=label,
            )
            descriptor = msvcrt.open_osfhandle(handle_owner.value, os.O_RDONLY)
            handle_owner.release()
            handle_owner = None
        else:
            if directory.posix_fd is None:
                raise SnapshotValidationError(f"{label} directory descriptor is closed")
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory.posix_fd,
            )
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise SnapshotValidationError(f"cannot inspect {label}: {name}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        elif handle_owner is not None:
            handle_owner.close()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or (int(metadata.st_dev), int(metadata.st_ino)) != expected_identity
    ):
        raise SnapshotValidationError(f"{label} identity changed: {name}")


def _rename_pinned_entry_no_replace(
    directory: _PinnedDirectory, source: str, target: str, *, label: str
) -> None:
    """Rename one POSIX entry relative to its pin without replacing a target."""
    import ctypes
    import errno

    if directory.posix_fd is None:
        raise SnapshotValidationError(f"{label} directory descriptor is closed")
    _require_leaf_name(source, label=label)
    _require_leaf_name(target, label=label)
    library = ctypes.CDLL(None, use_errno=True)
    operation: Any
    flags: int
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        operation = library.renameat2
        flags = 1  # RENAME_NOREPLACE
    elif sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        operation = library.renameatx_np
        flags = 0x00000004  # RENAME_EXCL
    else:
        raise SnapshotValidationError(
            f"{label} requires an atomic no-replace rename on this platform"
        )
    operation.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    operation.restype = ctypes.c_int
    result = operation(
        directory.posix_fd,
        os.fsencode(source),
        directory.posix_fd,
        os.fsencode(target),
        flags,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), target)
        raise OSError(error, os.strerror(error), source)


def _restore_quarantined_entry_after_mismatch(
    directory: _PinnedDirectory,
    quarantine: str,
    original: str,
    primary_error: BaseException,
    *,
    label: str,
) -> None:
    """Restore a post-rename substitution without replacing a concurrent name."""
    try:
        _rename_pinned_entry_no_replace(
            directory,
            quarantine,
            original,
            label=f"{label} restoration",
        )
    except BaseException as restoration_error:
        primary_error.add_note(f"{label} restoration also failed: {restoration_error}")
    raise primary_error


def _delete_pinned_file(
    directory: _PinnedDirectory,
    name: str,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    """Delete only the exact file created by this operation."""
    _require_leaf_name(name, label=label)
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        try:
            handle_owner = _windows_open_relative_handle(
                directory,
                name,
                desired_access=0x00010000 | 0x00000080 | 0x00100000,
                share_access=0x00000001 | 0x00000002,
                create_disposition=1,
                create_options=0x00000040 | 0x00000020 | 0x00200000,
                label=label,
            )
        except OSError as exc:
            if getattr(exc, "winerror", None) in (2, 3) or exc.errno in (2,):
                return
            raise SnapshotValidationError(f"could not open {label}: {name}") from exc
        descriptor: int | None = None
        try:
            descriptor = msvcrt.open_osfhandle(handle_owner.value, os.O_RDONLY)
            handle_owner.release()
            metadata = os.fstat(descriptor)
            observed = (int(metadata.st_dev), int(metadata.st_ino))
            if not stat.S_ISREG(metadata.st_mode) or observed != expected_identity:
                raise SnapshotValidationError(
                    f"refusing to remove substituted {label}: {name}"
                )

            class FileDispositionInfo(ctypes.Structure):
                _fields_ = [("DeleteFile", wintypes.BOOLEAN)]

            raw_handle = msvcrt.get_osfhandle(descriptor)
            disposition = FileDispositionInfo(True)
            set_information = ctypes.WinDLL(
                "kernel32", use_last_error=True
            ).SetFileInformationByHandle
            set_information.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                wintypes.LPVOID,
                wintypes.DWORD,
            ]
            set_information.restype = wintypes.BOOL
            if not set_information(
                raw_handle,
                4,  # FileDispositionInfo
                ctypes.byref(disposition),
                ctypes.sizeof(disposition),
            ):
                error = ctypes.get_last_error()
                raise SnapshotValidationError(
                    f"could not remove {label}: {name}"
                ) from OSError(error, os.strerror(error), str(directory.path / name))
        finally:
            if descriptor is not None:
                os.close(descriptor)
            else:
                handle_owner.close()
        return

    if directory.posix_fd is None:
        raise SnapshotValidationError(f"{label} directory descriptor is closed")
    quarantine = f".cleanup-{uuid.uuid4().hex}"
    try:
        _rename_pinned_entry_no_replace(
            directory,
            name,
            quarantine,
            label=f"{label} quarantine",
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SnapshotValidationError(f"could not quarantine {label}: {name}") from exc
    try:
        metadata = os.stat(quarantine, dir_fd=directory.posix_fd, follow_symlinks=False)
        observed = (int(metadata.st_dev), int(metadata.st_ino))
        if not stat.S_ISREG(metadata.st_mode) or observed != expected_identity:
            raise SnapshotValidationError(
                f"refusing to remove substituted {label}: {name}"
            )
    except BaseException as primary_error:
        _restore_quarantined_entry_after_mismatch(
            directory,
            quarantine,
            name,
            primary_error,
            label=label,
        )
    try:
        os.unlink(quarantine, dir_fd=directory.posix_fd)
    except OSError as exc:
        raise SnapshotValidationError(f"could not remove {label}: {name}") from exc


def _quarantine_pinned_directory(
    directory: _PinnedDirectory, parent: _PinnedDirectory, *, label: str
) -> str:
    """Detach an owned POSIX directory before traversal; Windows pins the name."""
    if os.name == "nt":
        return directory.path.name
    if directory.posix_fd is None or parent.posix_fd is None:
        raise SnapshotValidationError(f"{label} descriptor is closed")
    try:
        current = os.stat(
            directory.path.name,
            dir_fd=parent.posix_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise SnapshotValidationError(f"cannot inspect {label} name") from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or (int(current.st_dev), int(current.st_ino)) != directory.identity
    ):
        raise SnapshotValidationError(f"refusing to quarantine substituted {label}")
    quarantine = f".{directory.path.name}.cleanup-{uuid.uuid4().hex}"
    try:
        _rename_pinned_entry_no_replace(
            parent,
            directory.path.name,
            quarantine,
            label=f"{label} quarantine",
        )
    except OSError as exc:
        raise SnapshotValidationError(
            f"could not quarantine {label}: {directory.path}"
        ) from exc
    try:
        metadata = os.stat(quarantine, dir_fd=parent.posix_fd, follow_symlinks=False)
        observed = (int(metadata.st_dev), int(metadata.st_ino))
        if not stat.S_ISDIR(metadata.st_mode) or observed != directory.identity:
            raise SnapshotValidationError(f"refusing to clean substituted {label}")
    except BaseException as primary_error:
        _restore_quarantined_entry_after_mismatch(
            parent,
            quarantine,
            directory.path.name,
            primary_error,
            label=label,
        )
    return quarantine


def _purge_pinned_tree(directory: _PinnedDirectory, *, label: str) -> None:
    """Remove a private tree without following links or reopening its root."""
    if os.name != "nt":
        if directory.posix_fd is None:
            raise SnapshotValidationError(f"{label} descriptor is closed")
        for entry in list(os.scandir(directory.posix_fd)):
            metadata = entry.stat(follow_symlinks=False)
            identity = (int(metadata.st_dev), int(metadata.st_ino))
            if stat.S_ISDIR(metadata.st_mode):
                child = _open_pinned_child_directory(
                    directory,
                    entry.name,
                    directory.path / entry.name,
                    label=label,
                    expected_identity=identity,
                )
                quarantined_name: str | None = None
                try:
                    quarantined_name = _quarantine_pinned_directory(
                        child, directory, label=label
                    )
                    _purge_pinned_tree(child, label=label)
                    _remove_pinned_empty_directory(
                        child,
                        directory,
                        label=label,
                        entry_name=quarantined_name,
                    )
                finally:
                    _close_pinned_directory(child, label=label)
            elif stat.S_ISREG(metadata.st_mode):
                _delete_pinned_file(directory, entry.name, identity, label=label)
            else:
                raise SnapshotValidationError(f"{label} contains a non-regular entry")
        return

    for child_name in _windows_list_pinned_directory_names(directory, label=label):
        metadata = _windows_relative_entry_metadata(directory, child_name, label=label)
        if bool(
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise SnapshotValidationError(f"{label} contains a link or reparse point")
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if stat.S_ISDIR(metadata.st_mode):
            child = _open_pinned_child_directory(
                directory,
                child_name,
                directory.path / child_name,
                label=label,
                expected_identity=identity,
            )
            try:
                _purge_pinned_tree(child, label=label)
                _remove_pinned_empty_directory(child, directory, label=label)
            finally:
                _close_pinned_directory(child, label=label)
        elif stat.S_ISREG(metadata.st_mode):
            _delete_pinned_file(directory, child_name, identity, label=label)
        else:
            raise SnapshotValidationError(f"{label} contains a non-regular entry")


def _pinned_directory_names(directory: _PinnedDirectory, *, label: str) -> list[str]:
    if directory.posix_fd is not None:
        with os.scandir(directory.posix_fd) as entries:
            return [entry.name for entry in entries]
    return _windows_list_pinned_directory_names(directory, label=label)


def _pinned_directory_empty(directory: _PinnedDirectory) -> bool:
    return not _pinned_directory_names(directory, label="pinned directory")


def _require_pinned_inventory(
    directory: _PinnedDirectory,
    identities: Mapping[str, tuple[int, int]],
    expected_order: Sequence[str],
    *,
    label: str,
) -> None:
    """Bind the complete committed inventory through the retained directory pin."""
    if tuple(identities) != tuple(expected_order):
        raise SnapshotValidationError(f"{label} creation order is noncanonical")
    names = _pinned_directory_names(directory, label=label)
    if len(names) != len(set(names)) or set(names) != set(expected_order):
        raise SnapshotValidationError(f"{label} inventory is noncanonical")
    for name in expected_order:
        _require_pinned_file_identity(
            directory,
            name,
            identities[name],
            label=f"{label} {name}",
        )


def _retry_cleanup(operation: Any, *, label: str) -> None:
    last_error: Exception | None = None
    for attempt in range(_CLEANUP_ATTEMPTS):
        try:
            operation()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < _CLEANUP_ATTEMPTS:
                time.sleep(0.01 * (attempt + 1))
            continue
        return
    raise SnapshotValidationError(f"could not complete {label}") from last_error


def _finish_pinned_private_cleanup(
    directory: _PinnedDirectory,
    parent: _PinnedDirectory | None,
    file_identities: Mapping[str, tuple[int, int]],
    *,
    label: str,
    primary_error: BaseException | None,
) -> None:
    """Delete only builder-owned entries through the retained capture handles."""
    cleanup_error: Exception | None = None
    quarantined_name: str | None = None
    try:
        for name, identity in sorted(file_identities.items()):
            _retry_cleanup(
                lambda name=name, identity=identity: _delete_pinned_file(
                    directory, name, identity, label=f"{label} file"
                ),
                label=f"{label} file cleanup",
            )
        if os.name != "nt" and not _pinned_directory_empty(directory):
            raise SnapshotValidationError(f"{label} contains unowned entries")
        if os.name == "nt":
            _retry_cleanup(
                lambda: _remove_pinned_empty_directory(
                    directory, directory, label=label
                ),
                label=f"{label} directory cleanup",
            )
        else:
            if parent is None:
                raise SnapshotValidationError(f"{label} parent handle is closed")
            quarantined_name = _quarantine_pinned_directory(
                directory, parent, label=label
            )
            _retry_cleanup(
                lambda: _remove_pinned_empty_directory(
                    directory,
                    parent,
                    label=label,
                    entry_name=quarantined_name,
                ),
                label=f"{label} directory cleanup",
            )
    except Exception as exc:
        cleanup_error = exc
    finally:
        close_errors: list[Exception] = []
        pins = [(directory, label)]
        if parent is not None:
            pins.append((parent, f"{label} parent"))
        for pin, pin_label in pins:
            try:
                _close_pinned_directory(pin, label=pin_label)
            except Exception as exc:
                close_errors.append(exc)
        if cleanup_error is None and close_errors:
            cleanup_error = close_errors[0]
        elif cleanup_error is not None:
            for close_error in close_errors:
                cleanup_error.add_note(
                    f"{label} handle close also failed: {close_error}"
                )
    if cleanup_error is not None:
        if primary_error is None:
            raise cleanup_error
        primary_error.add_note(f"{label} cleanup also failed: {cleanup_error}")


@contextmanager
def _captured_raw_release(
    raw_root: Path, specs: Sequence[RawFileSpec]
) -> Iterator[tuple[list[dict[str, Any]], dict[str, CapturedFile]]]:
    raw_root = _absolute_without_resolving(raw_root)
    _reject_link_or_reparse_components(raw_root, label="Open Targets raw root")
    if not raw_root.is_dir():
        raise SnapshotValidationError(
            f"Open Targets {RELEASE} raw root is not a directory: {raw_root}"
        )
    expected = {spec.path for spec in specs}
    observed = {
        path.relative_to(raw_root).as_posix()
        for path in raw_root.rglob("*.parquet")
        if path.is_file()
    }
    if observed != expected:
        raise SnapshotValidationError(
            "Open Targets raw Parquet inventory differs from the fixed files: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )

    capture_root = Path(tempfile.mkdtemp(prefix="depplm-ot-capture-"))
    capture_pin = _pin_directory(
        capture_root, label="Open Targets private capture directory"
    )
    capture_parent_pin = (
        None
        if os.name == "nt"
        else _pin_directory(
            capture_root.parent,
            label="Open Targets private capture parent",
        )
    )
    records: list[dict[str, Any]] = []
    captures: dict[str, CapturedFile] = {}
    file_identities: dict[str, tuple[int, int]] = {}
    primary_error: BaseException | None = None
    try:
        for index, spec in enumerate(specs):
            relative = Path(spec.path)
            if relative.is_absolute() or any(
                part in ("", ".", "..") for part in relative.parts
            ):
                raise SnapshotValidationError(
                    "raw input path is not canonical relative"
                )
            source = raw_root / relative
            try:
                source.relative_to(raw_root)
            except ValueError as exc:
                raise SnapshotValidationError(
                    "raw input path escapes raw root"
                ) from exc
            captured = _capture_file_to_private(
                source,
                capture_root / f"{index:02d}.parquet",
                label=f"Open Targets {RELEASE} raw file",
                expected_bytes=spec.bytes,
                expected_sha256=spec.sha256,
                _destination_parent_pin=capture_pin,
            )
            file_identities[captured.path.name] = captured.identity
            captures[spec.path] = captured
            records.append(
                {"path": spec.path, "bytes": captured.bytes, "sha256": captured.sha256}
            )
        yield records, captures
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _finish_pinned_private_cleanup(
            capture_pin,
            capture_parent_pin,
            file_identities,
            label="Open Targets private capture directory",
            primary_error=primary_error,
        )


def _capture_small_bytes(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    capture_root = Path(tempfile.mkdtemp(prefix="depplm-snapshot-capture-"))
    capture_pin = _pin_directory(
        capture_root, label="snapshot metadata private capture directory"
    )
    capture_parent_pin = (
        None
        if os.name == "nt"
        else _pin_directory(
            capture_root.parent,
            label="snapshot metadata private capture parent",
        )
    )
    file_identities: dict[str, tuple[int, int]] = {}
    primary_error: BaseException | None = None
    try:
        payload_buffer = bytearray()
        captured = _capture_file_to_private(
            path,
            capture_root / "payload",
            label=label,
            _payload_sink=payload_buffer,
            _destination_parent_pin=capture_pin,
        )
        file_identities[captured.path.name] = captured.identity
        payload = bytes(payload_buffer)
        return payload, _record_for_bytes(payload, str(captured.source_path))
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _finish_pinned_private_cleanup(
            capture_pin,
            capture_parent_pin,
            file_identities,
            label="snapshot metadata private capture directory",
            primary_error=primary_error,
        )


def _safe_child(base: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SnapshotValidationError("snapshot path must be a non-empty string")
    raw = Path(relative)
    if raw.is_absolute() or any(part in ("", ".", "..") for part in raw.parts):
        raise SnapshotValidationError("snapshot path is not canonical relative")
    candidate = (base / raw).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise SnapshotValidationError("snapshot path escapes its root") from exc
    return candidate


def _iter_ids(value: Any) -> Iterable[str]:
    if value is None:
        return ()
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            value = value.tolist()
    except ImportError:
        pass
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("id", item.get("diseaseId"))
        if item is not None and str(item).strip():
            result.append(str(item).strip())
    return result


def classify_stage(stage: Any) -> str:
    normalized = str(stage).strip().upper() if stage is not None else ""
    if normalized in PHASE2_PLUS_STAGES:
        return "phase2_plus"
    if normalized == WITHDRAWAL_STAGE:
        return "withdrawal_sensitivity"
    return "excluded_stage"


def _complete_cancer_only_disease_ids(
    value: Any, cancer_ids: set[str]
) -> tuple[str, ...]:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            value = value.tolist()
    except ImportError:
        pass
    if not isinstance(value, (list, tuple)) or not value:
        return ()
    identifiers: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            return ()
        identifier = item.get("diseaseId", item.get("id"))
        if identifier is None or not str(identifier).strip():
            return ()
        identifiers.append(str(identifier).strip())
    if any(identifier not in cancer_ids for identifier in identifiers):
        return ()
    return tuple(sorted(set(identifiers)))


def cancer_disease_scope(
    disease_rows: Sequence[Mapping[str, Any]], root: str = CANCER_ROOT
) -> tuple[set[str], list[dict[str, str]]]:
    root_rows = [row for row in disease_rows if str(row.get("id", "")) == root]
    if len(root_rows) != 1:
        raise SnapshotValidationError(
            f"expected exactly one disease row for cancer root {root}"
        )
    root_row = root_rows[0]
    identifiers = {root, *set(_iter_ids(root_row.get("descendants")))}
    for row in disease_rows:
        identifier = str(row.get("id", "")).strip()
        if identifier and root in set(_iter_ids(row.get("ancestors"))):
            identifiers.add(identifier)
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in disease_rows:
        identifier = str(row.get("id", "")).strip()
        if identifier in by_id:
            raise SnapshotValidationError(f"duplicate disease identifier: {identifier}")
        if identifier:
            by_id[identifier] = row
    if identifiers - set(by_id):
        raise SnapshotValidationError("cancer descendants are absent from disease data")
    direct = set(_iter_ids(root_row.get("descendants")))
    scope = []
    for identifier in sorted(identifiers):
        relation = "root" if identifier == root else "descendant"
        if identifier != root and identifier not in direct:
            relation = "ancestor_relation"
        scope.append(
            {
                "disease_id": identifier,
                "name": str(by_id[identifier].get("name") or ""),
                "scope_relation": relation,
            }
        )
    return identifiers, scope


def build_snapshot_rows(
    target_rows: Sequence[Mapping[str, Any]],
    disease_rows: Sequence[Mapping[str, Any]],
    clinical_rows: Sequence[Mapping[str, Any]],
    tractability_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]], dict[str, Any]
]:
    cancer_ids, disease_scope = cancer_disease_scope(disease_rows)
    symbol_to_ids: dict[str, set[str]] = defaultdict(set)
    id_to_symbols: dict[str, set[str]] = defaultdict(set)
    for row in target_rows:
        identifier = str(row.get("id") or "").strip()
        symbol = str(row.get("approvedSymbol") or "").strip()
        if identifier and symbol:
            symbol_to_ids[symbol].add(identifier)
            id_to_symbols[identifier].add(symbol)
    unambiguous = {
        symbol: next(iter(identifiers))
        for symbol, identifiers in symbol_to_ids.items()
        if len(identifiers) == 1 and len(id_to_symbols[next(iter(identifiers))]) == 1
    }
    ambiguous = [
        {
            "approved_symbol": symbol,
            "target_ids": ";".join(sorted(identifiers)),
            "reason": "symbol_or_target_not_one_to_one",
        }
        for symbol, identifiers in sorted(symbol_to_ids.items())
        if len(identifiers) != 1
        or any(len(id_to_symbols[identifier]) != 1 for identifier in identifiers)
    ]
    stages_by_target: dict[str, set[str]] = defaultdict(set)
    drugs_by_target: dict[str, set[str]] = defaultdict(set)
    withdrawals: dict[str, set[str]] = defaultdict(set)
    rows_with_cancer = cancer_only_rows = mixed_rows = unmapped_rows = 0
    for row in clinical_rows:
        target_id = str(row.get("targetId") or "").strip()
        indications = set(_iter_ids(row.get("diseases")))
        if not target_id or not indications.intersection(cancer_ids):
            continue
        rows_with_cancer += 1
        raw_diseases = row.get("diseases")
        try:
            import numpy as np

            if isinstance(raw_diseases, np.ndarray):
                raw_diseases = raw_diseases.tolist()
        except ImportError:
            pass
        if not isinstance(raw_diseases, (list, tuple)) or any(
            not isinstance(item, Mapping)
            or not str(item.get("diseaseId", item.get("id")) or "").strip()
            for item in raw_diseases
        ):
            unmapped_rows += 1
            continue
        if any(identifier not in cancer_ids for identifier in indications):
            mixed_rows += 1
            continue
        if not _complete_cancer_only_disease_ids(raw_diseases, cancer_ids):
            raise SnapshotValidationError("cancer-only classification is inconsistent")
        cancer_only_rows += 1
        stage = str(row.get("maxClinicalStage") or "").strip().upper()
        drug_id = str(row.get("drugId") or "").strip()
        classification = classify_stage(stage)
        if classification == "phase2_plus":
            stages_by_target[target_id].add(stage)
            if drug_id:
                drugs_by_target[target_id].add(drug_id)
        elif classification == "withdrawal_sensitivity" and drug_id:
            withdrawals[target_id].add(drug_id)
    modalities: dict[str, set[str]] = defaultdict(set)
    tractability_counts: Counter[str] = Counter()
    for row in tractability_rows:
        if row.get("value") is not True:
            continue
        target_id = str(row.get("targetId") or "").strip()
        modality = str(row.get("modality") or "").strip()
        if target_id:
            tractability_counts[target_id] += 1
            if modality:
                modalities[target_id].add(modality)
    outcomes: list[dict[str, Any]] = []
    for symbol, target_id in sorted(unambiguous.items()):
        stages = stages_by_target.get(target_id, set())
        outcomes.append(
            {
                "approved_symbol": symbol,
                "target_id": target_id,
                "clinical_oncology_phase2_plus": int(bool(stages)),
                "oncology_phase2_plus_drug_count": len(drugs_by_target[target_id]),
                "oncology_phase2_plus_stages": ";".join(
                    sorted(stages, key=STAGE_ORDER.__getitem__)
                ),
                "maximum_oncology_phase2_plus_stage": (
                    max(stages, key=STAGE_ORDER.__getitem__) if stages else ""
                ),
                "clinical_oncology_withdrawal": int(bool(withdrawals[target_id])),
                "oncology_withdrawn_drug_count": len(withdrawals[target_id]),
                "tractable_any": int(bool(tractability_counts[target_id])),
                "tractability_true_record_count": int(tractability_counts[target_id]),
                "tractability_true_modalities": ";".join(sorted(modalities[target_id])),
            }
        )
    counts = {
        "target_rows": len(target_rows),
        "disease_rows": len(disease_rows),
        "clinical_target_rows": len(clinical_rows),
        "tractability_rows": len(tractability_rows),
        "cancer_disease_count": len(cancer_ids),
        "rows_with_any_cancer_disease": rows_with_cancer,
        "cancer_only_clinical_rows": cancer_only_rows,
        "mixed_cancer_noncancer_rows_excluded": mixed_rows,
        "cancer_rows_with_unmapped_disease_excluded": unmapped_rows,
        "symbols_observed": len(symbol_to_ids),
        "one_to_one_symbols": len(unambiguous),
        "ambiguous_symbols_excluded": len(ambiguous),
        "clinical_oncology_phase2_plus_targets": sum(
            int(row["clinical_oncology_phase2_plus"]) for row in outcomes
        ),
        "clinical_oncology_withdrawal_targets": sum(
            int(row["clinical_oncology_withdrawal"]) for row in outcomes
        ),
        "tractable_targets": sum(int(row["tractable_any"]) for row in outcomes),
    }
    return outcomes, disease_scope, ambiguous, counts


def _tsv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(fields), delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def _read_parquet_rows(
    paths: Sequence[Path], columns: Sequence[str]
) -> list[dict[str, Any]]:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise SnapshotValidationError("snapshot validation requires pyarrow") from exc
    try:
        table = ds.dataset([str(path) for path in paths], format="parquet").to_table(
            columns=list(columns)
        )
    except Exception as exc:
        raise SnapshotValidationError(
            f"cannot read fixed Parquet inputs: {exc}"
        ) from exc
    return table.to_pylist()


def validate_raw_release(
    raw_root: Path, specs: Sequence[RawFileSpec] = RAW_FILES
) -> list[dict[str, Any]]:
    with _captured_raw_release(raw_root, specs) as (records, _captures):
        return records


def _require_registered_specs(specs: Sequence[RawFileSpec]) -> None:
    observed = tuple((spec.path, spec.bytes, spec.sha256) for spec in specs)
    expected = tuple((spec.path, spec.bytes, spec.sha256) for spec in RAW_FILES)
    if observed != expected:
        raise SnapshotValidationError(
            "snapshot build specification differs from the registered Open Targets "
            f"{RELEASE} raw-file inventory"
        )


def _derive_snapshot_from_captures(
    captures: Mapping[str, CapturedFile], specs: Sequence[RawFileSpec]
) -> tuple[
    list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]], dict[str, Any]
]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for spec in specs:
        captured = captures.get(spec.path)
        if captured is None:
            raise SnapshotValidationError("raw capture inventory is incomplete")
        grouped[spec.path.split("/", 1)[0]].append(captured.path)
    required = {"target", "clinical_target", "disease", "target_tractability"}
    if set(grouped) != required:
        raise SnapshotValidationError("raw specification groups are noncanonical")
    return build_snapshot_rows(
        _read_parquet_rows(grouped["target"], ("id", "approvedSymbol")),
        _read_parquet_rows(
            grouped["disease"], ("id", "name", "ancestors", "descendants")
        ),
        _read_parquet_rows(
            grouped["clinical_target"],
            ("targetId", "drugId", "diseases", "maxClinicalStage"),
        ),
        _read_parquet_rows(
            grouped["target_tractability"],
            ("targetId", "modality", "category", "value"),
        ),
    )


def _validate_freeze_request(
    raw_root: Path, output_dir: Path, specs: Sequence[RawFileSpec]
) -> tuple[Path, Path]:
    _require_registered_specs(specs)
    raw_root = _absolute_without_resolving(raw_root)
    output_dir = _absolute_without_resolving(output_dir)
    _reject_link_or_reparse_components(
        output_dir.parent, label="snapshot output parent"
    )
    if os.path.lexists(output_dir):
        raise SnapshotValidationError(
            f"refusing to overwrite snapshot directory: {output_dir}"
        )
    return raw_root, output_dir


def _reserve_snapshot_publication(output_dir: Path) -> _PendingSnapshotPublication:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_or_reparse_components(
        output_dir.parent, label="snapshot output parent"
    )
    parent_pin = _pin_directory(output_dir.parent, label="snapshot output parent")
    parent_identity = parent_pin.identity
    try:
        staging = output_dir
        staging_pin = _create_pinned_directory(
            parent_pin,
            output_dir.name,
            staging,
            label="reserved snapshot destination",
        )
    except FileExistsError as exc:
        try:
            _close_pinned_directory(parent_pin, label="snapshot output parent")
        except Exception as cleanup_error:
            exc.add_note(f"snapshot parent-handle cleanup failed: {cleanup_error}")
        raise SnapshotValidationError(
            f"refusing to overwrite snapshot directory: {output_dir}"
        ) from exc
    except BaseException as exc:
        try:
            _close_pinned_directory(parent_pin, label="snapshot output parent")
        except Exception as cleanup_error:
            exc.add_note(f"snapshot parent-handle cleanup failed: {cleanup_error}")
        raise
    return _PendingSnapshotPublication(
        destination=output_dir,
        staging=staging,
        parent_identity=parent_identity,
        staging_identity=staging_pin.identity,
        parent_pin=parent_pin,
        staging_pin=staging_pin,
    )


def _close_snapshot_publication_pins(
    publication: _PendingSnapshotPublication,
    *,
    primary_error: BaseException | None = None,
) -> None:
    close_errors: list[Exception] = []
    for pin, label in (
        (publication.staging_pin, "reserved snapshot destination"),
        (publication.parent_pin, "snapshot output parent"),
    ):
        try:
            _close_pinned_directory(pin, label=label)
        except Exception as exc:
            close_errors.append(exc)
    if close_errors:
        for error in close_errors:
            if primary_error is not None:
                primary_error.add_note(
                    f"publication-handle cleanup also failed: {error}"
                )
        if primary_error is None:
            raise close_errors[0]


def _remove_pinned_empty_directory(
    directory: _PinnedDirectory,
    parent: _PinnedDirectory,
    *,
    label: str,
    entry_name: str | None = None,
) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        if directory.windows_handle is None:
            raise SnapshotValidationError(f"{label} handle is closed")

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = [("DeleteFile", wintypes.BOOLEAN)]

        disposition = FileDispositionInfo(True)
        set_information = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).SetFileInformationByHandle
        set_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        set_information.restype = wintypes.BOOL
        if not set_information(
            directory.windows_handle.value,
            4,  # FileDispositionInfo
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            error = ctypes.get_last_error()
            raise SnapshotValidationError(
                f"could not remove {label}: {directory.path}"
            ) from OSError(error, os.strerror(error), str(directory.path))
        # Windows completes deletion only after the final handle closes.  Close
        # this child now so a retained parent can subsequently become empty.
        _close_pinned_directory(directory, label=label)
        return

    if parent.posix_fd is None:
        raise SnapshotValidationError("snapshot parent descriptor is closed")
    try:
        os.rmdir(
            directory.path.name if entry_name is None else entry_name,
            dir_fd=parent.posix_fd,
        )
    except OSError as exc:
        raise SnapshotValidationError(
            f"could not remove {label}: {directory.path}"
        ) from exc


def _cleanup_owned_snapshot_destination(
    publication: _PendingSnapshotPublication,
) -> None:
    primary_error: BaseException | None = None
    quarantined_name: str | None = None
    try:
        for name, identity in list(publication.created_file_identities.items()):
            _delete_pinned_file(
                publication.staging_pin,
                name,
                identity,
                label="owned snapshot artifact",
            )
        quarantined_name = _quarantine_pinned_directory(
            publication.staging_pin,
            publication.parent_pin,
            label="reserved snapshot destination",
        )
        _remove_pinned_empty_directory(
            publication.staging_pin,
            publication.parent_pin,
            label="reserved snapshot destination",
            entry_name=quarantined_name,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _close_snapshot_publication_pins(publication, primary_error=primary_error)


def _publish_snapshot_directory(publication: _PendingSnapshotPublication) -> Path:
    """Commit the pinned inventory and return a point-in-time validated path.

    The final-named directory is visible while it is populated. Publication is
    commit-marker based, not an atomic directory rename, so consumers must
    reject a missing or invalid manifest marker.

    A returned ``Path`` is not a retained capability.  Downstream consumers
    must independently verify the committed file hashes; this function does
    not claim protection from same-UID replacement after its final check.
    """
    _require_pinned_directory_entry(
        publication.parent_pin,
        publication.destination.name,
        publication.staging_identity,
        label="reserved snapshot destination",
    )
    if "snapshot-manifest.json" not in publication.created_file_identities:
        raise SnapshotValidationError("snapshot has no manifest commit marker")
    expected_order = (*ARTIFACT_FIELDS, "snapshot-manifest.json")
    _require_pinned_inventory(
        publication.staging_pin,
        publication.created_file_identities,
        expected_order,
        label="snapshot publication",
    )
    _close_snapshot_publication_pins(publication)
    _require_directory_identity(
        publication.destination,
        publication.staging_identity,
        label="published snapshot destination",
    )
    return publication.destination


def _protocol_sections() -> dict[str, Any]:
    return {
        "cancer_scope": {
            "root": CANCER_ROOT,
            "rule": "root plus descendants and rows whose ancestors contain the root",
        },
        "clinical_endpoint": {
            "source_table": "clinical_target",
            "cancer_only_complete_disease_list_required": True,
            "unmapped_or_mixed_disease_rows_excluded": True,
            "phase2_plus_stages": list(PHASE2_PLUS_STAGES),
            "maxClinicalStage_interpretation": (
                "historical row-level maximum; not current activity"
            ),
            "withdrawal_stage": WITHDRAWAL_STAGE,
            "withdrawal_role": "separate_sensitivity_only",
            "associatedTargets_used": False,
        },
        "tractability_endpoint": {
            "rule": (
                "target has at least one target_tractability record with value exactly true"
            ),
            "false_or_null_records_count_as_positive": False,
        },
        "target_mapping": {
            "rule": (
                "case-sensitive exact approvedSymbol with one symbol to one target id"
            ),
            "ambiguous_symbols_excluded": True,
        },
    }


def _builder_record() -> dict[str, Any]:
    import pyarrow

    source_record = {
        "path": _LOADED_SOURCE_PATH.relative_to(ROOT.resolve()).as_posix(),
        **_LOADED_SOURCE_FINGERPRINT,
    }
    return {
        **source_record,
        "builder_api": BUILDER_API,
        "python": platform.python_version(),
        "pyarrow": pyarrow.__version__,
    }


def preflight_freeze(
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    specs: Sequence[RawFileSpec] = RAW_FILES,
) -> dict[str, Any]:
    _require_loaded_source_unchanged()
    raw_root, output_dir = _validate_freeze_request(raw_root, output_dir, specs)
    records = validate_raw_release(raw_root, specs)
    return {
        "schema": SNAPSHOT_SCHEMA,
        "status": "ready",
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "raw_files_rehashed": len(records),
        "builder": _builder_record(),
        "freeze_command_argv": [
            "python",
            "code/clinical_snapshot_builder_v4.py",
            "freeze",
            "--raw-root",
            str(raw_root),
            "--output-dir",
            str(output_dir),
        ],
    }


def freeze_snapshot(
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    specs: Sequence[RawFileSpec] = RAW_FILES,
) -> Path:
    """Build a snapshot and return its point-in-time validated manifest path.

    The returned ``Path`` is not an identity-bearing capability.  A downstream
    consumer must reopen the directory and call :func:`validate_snapshot`
    immediately before use; that verifier rederives the registered inventory,
    hashes, and commit marker from the raw release.
    """
    _require_loaded_source_unchanged()
    raw_root, output_dir = _validate_freeze_request(raw_root, output_dir, specs)
    publication = _reserve_snapshot_publication(output_dir)
    primary_error: BaseException | None = None
    try:
        with _captured_raw_release(raw_root, specs) as (raw_records, captures):
            outcomes, disease_scope, ambiguous, counts = _derive_snapshot_from_captures(
                captures, specs
            )
            rows_by_name = {
                "target_outcomes.tsv": outcomes,
                "disease_scope.tsv": disease_scope,
                "ambiguous_symbols.tsv": ambiguous,
            }
            artifact_records = []
            for name, fields in ARTIFACT_FIELDS.items():
                payload = _tsv_bytes(rows_by_name[name], fields)
                _write_pinned_bytes(
                    publication.staging_pin,
                    name,
                    payload,
                    publication.created_file_identities,
                    label="snapshot artifact",
                )
                artifact_records.append(_record_for_bytes(payload, name))
            manifest = {
                "schema": SNAPSHOT_SCHEMA,
                "release": RELEASE,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "scientific_result": False,
                "scientific_result_reason": SNAPSHOT_RESULT_REASON,
                **_protocol_sections(),
                "raw_files": raw_records,
                "artifacts": artifact_records,
                "counts": counts,
                "builder": _builder_record(),
            }
            _write_pinned_bytes(
                publication.staging_pin,
                "snapshot-manifest.json",
                _canonical_json_bytes(manifest),
                publication.created_file_identities,
                label="snapshot manifest commit marker",
            )
        _publish_snapshot_directory(publication)
    except BaseException as exc:
        primary_error = exc
        try:
            _cleanup_owned_snapshot_destination(publication)
        except Exception as cleanup_error:
            primary_error.add_note(
                "snapshot staging cleanup also failed: " f"{cleanup_error}"
            )
        raise
    return output_dir / "snapshot-manifest.json"


def _validate_builder_binding(manifest: Mapping[str, Any]) -> None:
    builder = manifest.get("builder")
    if not isinstance(builder, dict) or set(builder) != {
        "path",
        "bytes",
        "sha256",
        "builder_api",
        "python",
        "pyarrow",
    }:
        raise SnapshotValidationError("snapshot builder binding is noncanonical")
    if builder.get("path") != "code/clinical_snapshot_builder_v4.py":
        raise SnapshotValidationError("snapshot builder path is noncanonical")
    if builder.get("builder_api") != BUILDER_API:
        raise SnapshotValidationError("snapshot builder API is noncanonical")
    if not all(
        isinstance(builder.get(field), str) and builder[field]
        for field in ("python", "pyarrow")
    ):
        raise SnapshotValidationError("snapshot builder environment is incomplete")
    import pyarrow

    if builder.get("python") != platform.python_version():
        raise SnapshotValidationError("snapshot builder Python runtime changed")
    if builder.get("pyarrow") != pyarrow.__version__:
        raise SnapshotValidationError("snapshot builder PyArrow runtime changed")
    source_record = _LOADED_SOURCE_FINGERPRINT
    if source_record["bytes"] != builder["bytes"]:
        raise SnapshotValidationError("snapshot builder source byte count changed")
    if source_record["sha256"] != builder["sha256"]:
        raise SnapshotValidationError("snapshot builder source SHA-256 changed")


def _validate_legacy_generator(manifest: Mapping[str, Any]) -> None:
    generator = manifest.get("generator")
    if not isinstance(generator, dict) or set(generator) != {
        "path",
        "bytes",
        "sha256",
        "python",
        "pyarrow",
    }:
        raise SnapshotValidationError("legacy snapshot generator is noncanonical")
    for key, expected in LEGACY_V3_GENERATOR.items():
        if generator.get(key) != expected:
            raise SnapshotValidationError(
                f"legacy snapshot generator {key} is not the registered anchor"
            )
    normalized = str(generator.get("path", "")).replace("\\", "/")
    if not normalized.endswith("/code/clinical_validation_v2.py"):
        raise SnapshotValidationError("legacy snapshot generator path is noncanonical")


def validate_snapshot(
    snapshot_dir: Path,
    *,
    specs: Sequence[RawFileSpec] = RAW_FILES,
    raw_root: Path | None = None,
) -> dict[str, Any]:
    """Rederive a snapshot and validate the bytes visible during this call.

    This is a point-in-time verification.  It does not turn ``snapshot_dir``
    into a retained capability or protect later pathname use from a same-UID
    replacement.
    """
    _require_loaded_source_unchanged()
    _require_registered_specs(specs)
    snapshot_dir = snapshot_dir.resolve()
    manifest_path = snapshot_dir / "snapshot-manifest.json"
    manifest_payload, manifest_record = _capture_small_bytes(
        manifest_path, label="snapshot manifest"
    )
    manifest = _strict_json_bytes(manifest_payload, label="snapshot manifest")
    if not isinstance(manifest, dict) or manifest.get("schema") not in {
        SNAPSHOT_SCHEMA,
        LEGACY_SNAPSHOT_SCHEMA,
    }:
        raise SnapshotValidationError("snapshot manifest schema is invalid")
    if manifest.get(
        "schema"
    ) == SNAPSHOT_SCHEMA and manifest_payload != _canonical_json_bytes(manifest):
        raise SnapshotValidationError(
            "v4 snapshot manifest bytes are not canonical JSON"
        )
    source_field = "builder" if manifest["schema"] == SNAPSHOT_SCHEMA else "generator"
    expected_fields = {
        "schema",
        "release",
        "created_utc",
        "scientific_result",
        "scientific_result_reason",
        "cancer_scope",
        "clinical_endpoint",
        "tractability_endpoint",
        "target_mapping",
        "raw_files",
        "artifacts",
        "counts",
        source_field,
    }
    if set(manifest) != expected_fields:
        raise SnapshotValidationError("snapshot manifest fields are noncanonical")
    if (
        manifest.get("release") != RELEASE
        or manifest.get("scientific_result") is not False
        or manifest.get("scientific_result_reason") != SNAPSHOT_RESULT_REASON
    ):
        raise SnapshotValidationError("snapshot release/result boundary is invalid")
    _require_canonical_utc_timestamp(
        manifest.get("created_utc"), label="snapshot created_utc"
    )
    for key, expected in _protocol_sections().items():
        if manifest.get(key) != expected:
            raise SnapshotValidationError(f"snapshot {key} is noncanonical")
    expected_raw = [
        {"path": spec.path, "bytes": spec.bytes, "sha256": spec.sha256}
        for spec in specs
    ]
    if manifest.get("raw_files") != expected_raw:
        raise SnapshotValidationError("snapshot raw-file bindings differ from release")
    bound_raw_root = _absolute_without_resolving(
        raw_root if raw_root is not None else snapshot_dir.parent / "raw"
    )
    with _captured_raw_release(bound_raw_root, specs) as (_raw_records, captures):
        outcomes, disease_scope, ambiguous, counts = _derive_snapshot_from_captures(
            captures, specs
        )
        if manifest.get("counts") != counts:
            raise SnapshotValidationError(
                "snapshot counts differ from raw-data rederivation"
            )
        rows_by_name = {
            "target_outcomes.tsv": outcomes,
            "disease_scope.tsv": disease_scope,
            "ambiguous_symbols.tsv": ambiguous,
        }
        records = manifest.get("artifacts")
        if not isinstance(records, list) or len(records) != len(ARTIFACT_FIELDS):
            raise SnapshotValidationError("snapshot artifact inventory is invalid")
        names = [
            record.get("path") if isinstance(record, dict) else None
            for record in records
        ]
        if tuple(names) != tuple(ARTIFACT_FIELDS):
            raise SnapshotValidationError("snapshot artifact names are noncanonical")
        expected_inventory = {*ARTIFACT_FIELDS, "snapshot-manifest.json"}
        entries = list(snapshot_dir.iterdir())
        if {entry.name for entry in entries} != expected_inventory or any(
            not entry.is_file()
            or entry.is_symlink()
            or bool(
                getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
            for entry in entries
        ):
            raise SnapshotValidationError(
                "snapshot directory inventory is noncanonical"
            )
        for record in records:
            if not isinstance(record, dict):
                raise SnapshotValidationError("snapshot artifact record is invalid")
            path = _safe_child(snapshot_dir, record["path"])
            payload, observed_record = _capture_small_bytes(
                path, label="snapshot artifact"
            )
            expected_record = {
                "path": record.get("path"),
                "bytes": record.get("bytes"),
                "sha256": record.get("sha256"),
            }
            if (
                set(record) != set(expected_record)
                or {
                    "path": record.get("path"),
                    "bytes": observed_record["bytes"],
                    "sha256": observed_record["sha256"],
                }
                != expected_record
            ):
                raise SnapshotValidationError(
                    f"snapshot artifact binding changed: {record.get('path')}"
                )
            expected = _tsv_bytes(
                rows_by_name[record["path"]], ARTIFACT_FIELDS[record["path"]]
            )
            if payload != expected:
                raise SnapshotValidationError(
                    "snapshot artifact differs from raw-data rederivation: "
                    f"{record['path']}"
                )
    if manifest["schema"] == SNAPSHOT_SCHEMA:
        _validate_builder_binding(manifest)
    else:
        _validate_legacy_generator(manifest)
    return {
        "manifest": manifest,
        "manifest_record": manifest_record,
        "snapshot_dir": str(snapshot_dir),
        "raw_files_rehashed": len(expected_raw),
        "artifacts_rederived": len(ARTIFACT_FIELDS),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("preflight", "rehash canonical raw inputs without creating a snapshot"),
        ("freeze", "create a new source-hash-bound builder-v4 snapshot"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
        child.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    verify = subparsers.add_parser(
        "verify", help="rederive and verify an existing v3 or v4 snapshot"
    )
    verify.add_argument("--snapshot-dir", type=Path, required=True)
    verify.add_argument(
        "--raw-root",
        type=Path,
        help="raw input root; defaults to the snapshot directory's sibling raw/",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "preflight":
            print(
                json.dumps(
                    preflight_freeze(args.raw_root, args.output_dir),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "freeze":
            print(freeze_snapshot(args.raw_root, args.output_dir))
        elif args.command == "verify":
            result = validate_snapshot(args.snapshot_dir, raw_root=args.raw_root)
            print(json.dumps(result["manifest_record"], sort_keys=True))
        else:
            raise AssertionError(f"unsupported command: {args.command}")
    except (SnapshotValidationError, OSError, ValueError, KeyError) as exc:
        print(f"[clinical-snapshot-builder-v4] REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
