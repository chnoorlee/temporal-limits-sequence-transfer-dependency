"""Fixed-protocol Sanger organoid validation for DepPLM.

This is a documented post-download revision after a pre-numeric-access workflow
failure; it is not a prospective preregistration.  The CLI enforces distinct
artifact phases:

* ``audit-features`` validates only non-outcome covariates, RNA and driver data.
* ``freeze`` creates a write-once fixed-protocol lock without an outcome argument.
* ``lock-predictions`` validates the freeze and writes a prediction lock without an
  outcome argument.
* ``analyze`` requires externally retained freeze and prediction-lock SHA-256
  values before it can copy or parse the publisher outcome file.
* ``verify`` requires those two hashes plus an externally retained completion
  SHA-256 and performs deterministic analysis replay from the locked predictions
  plus integrity checks; it does not retrain models or regenerate predictions.

No workflow CLI subcommand downloads data.  Phase-local flags describe only what
that command accepts and reads; they are not independent evidence of real-world
chronology or of what a person or another process may have accessed.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import ctypes
import errno
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
import re
import shutil
import stat as stat_module
import statistics
import subprocess
import sys
import sysconfig
import threading
import types
import uuid
from builtins import BaseExceptionGroup
from collections import Counter
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

ROOT = Path(__file__).resolve().parent.parent
_CONTROLLER_SOURCE_RELATIVE = "code/organoid_independent_validation.py"
_LAUNCHER_SOURCE_RELATIVE = "code/run_organoid_independent_validation.py"
_CONTROLLER_RECORD_KEY = "_ORGANOID_CONTROLLER_EXECUTED_SOURCE_RECORD"
_CONTROLLER_PATH_KEY = "_ORGANOID_CONTROLLER_EXECUTED_SOURCE_PATH"
_LAUNCHER_RECORD_KEY = "_ORGANOID_LAUNCHER_EXECUTED_SOURCE_RECORD"
_LAUNCHER_PATH_KEY = "_ORGANOID_LAUNCHER_EXECUTED_SOURCE_PATH"
FORMAL_LAUNCH_COMMAND = (
    r".\.venv\Scripts\python.exe -I code\run_organoid_independent_validation.py"
)
PROTOCOL_PATH = (
    ROOT / "docs" / "revision_v2" / "ORGANOID_INDEPENDENT_VALIDATION_PROTOCOL.md"
)
DEFAULT_FEATURE_DIR = (
    ROOT / "data" / "external" / "sanger_organoid_2026" / "features-v1"
)
DEFAULT_GENES = ROOT / "data" / "processed" / "genes.txt"
DEFAULT_CAMPAIGN = (
    ROOT
    / "results"
    / "corrected_v2"
    / "campaigns"
    / "20261020T230939.640594900Z-campaign-c41b987c"
    / "campaign.json"
)
DEFAULT_FREEZE = (
    ROOT / "results" / "organoid_independent_validation_v1" / "protocol-lock-rev2.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "organoid_independent_validation_v1"

FEATURE_INPUTS = {
    "covariates": {
        "id": 63130933,
        "name": "data_covariates_ALL-organoids.csv",
        "bytes": 7_770,
        "md5": "2787e7ab6c64e4a9d6af9eef333ac936",
    },
    "driver_mutations": {
        "id": 63130939,
        "name": "data_driver-mutations_ALL-organoids.csv",
        "bytes": 136_771,
        "md5": "5a2856b2f3c2e0d05a09fa98f4a32a57",
    },
    "driver_variants": {
        "id": 63130942,
        "name": "data_driver-variants_ALL-organoids.csv",
        "bytes": 17_261,
        "md5": "9ac5f560cc5ed3a448bb4e5c0dd3610a",
    },
    "rna": {
        "id": 63130957,
        "name": "data_RNAseq_ALL-organoids.csv",
        "bytes": 65_275_850,
        "md5": "2642cc4aaffeb84f8add28417343760a",
    },
}
FEATURE_FILES = {
    **{label: str(record["name"]) for label, record in FEATURE_INPUTS.items()},
    "figshare_metadata": "figshare-article-28339340.json",
}
FEATURE_FILE_IDS = {
    label: int(record["id"]) for label, record in FEATURE_INPUTS.items()
}
OUTCOME_FILES = {
    "primary_qnorm_lfc": {
        "id": 63191101,
        "name": "07_EssMatrix_qnorm_corrected_logFCs.RData",
        "bytes": 125_803_583,
        "md5": "84b5e153c267dfa5ebd22a6d5e43a42a",
    },
    "unused_binary_depletion": {
        "id": 63130879,
        "name": "07_EssMatrix_bDepletionsB2.Rdata",
        "bytes": 2_826_657,
        "md5": "b9e700bde7e9debc1c7338c7460f6944",
    },
}
FIGSHARE_ITEM_ID = 28339340
FIGSHARE_DOI = "10.6084/m9.figshare.28339340.v1"
OFFICIAL_REPOSITORY_COMMIT = "0acbbd498f686627960deac08ee2f639af67f377"

CONDITIONS = ("DepPLM", "C-PLM-MLP")
SEEDS = (42, 43, 44)
FOLDS = tuple(f"fold-{index:02d}" for index in range(5))
LINEAGE_MAP = {
    "COLO": "Bowel",
    "OESO": "Esophagus/Stomach",
    "PANC": "Pancreas",
}
EXCLUDED_SAMPLE_LIBRARIES = (
    ("HCM-SANG-0267-D12", "minLib"),
    ("HCM-SANG-0268-C18", "v1_1"),
    ("HCM-SANG-0270-C20", "minLib"),
    ("HCM-SANG-0272-C20", "v1_1"),
    ("HCM-SANG-0273-C18", "minLib"),
    ("HCM-SANG-0276-C18", "minLib"),
    ("HCM-SANG-0282-C18", "minLib"),
    ("HCM-SANG-0529-C18", "v1_1"),
    ("HCM-SANG-0284-C18", "minLib"),
    ("HCM-SANG-0288-C18", "v1_1"),
    ("WTSI-COLO_278", "minLib"),
    ("WTSI-COLO_376", "minLib"),
    ("HCM-SANG-0291-C15", "v1_1"),
    ("HCM-SANG-0299-C15-A", "minLib"),
    ("HCM-SANG-0299-C15-B", "minLib"),
    ("HCM-SANG-0306-C15", "minLib"),
)

MIN_RNA_SAMPLES = 200
MIN_RNA_GENE_OVERLAP = 10_000
MIN_PRIMARY_ORGANOIDS = 100
MIN_PRIMARY_GENES = 5_000
MIN_GENES_PER_ORGANOID = 5_000
MIN_ORGANOIDS_PER_GENE = 20
TOP_FRACTION = 0.05
BOOTSTRAP_REPLICATES = 10_000
SIGN_FLIP_MONTE_CARLO = 100_000
INFERENCE_SEED = 20_261_022
PYREADR_REQUIRED_VERSION = "0.5.6"
REQUIRED_RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "torch",
    "pyreadr",
)
RUNTIME_MODULE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scipy.stats": "scipy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "pyreadr": "pyreadr",
    "pyreadr._pyreadr_parser": "pyreadr",
    "pyreadr.librdata": "pyreadr",
}

AUDIT_SCHEMA = "DepPLM-Sanger-organoid-feature-audit-v2"
FREEZE_SCHEMA = "DepPLM-Sanger-organoid-protocol-lock-v2"
PREDICTION_LOCK_SCHEMA = "DepPLM-Sanger-organoid-prediction-lock-v2"
TRAINING_RECORDS_SCHEMA = "DepPLM-Sanger-organoid-full-refit-records-v2"
SOURCE_BINDINGS_SCHEMA = "DepPLM-Sanger-organoid-source-bindings-v2"
ANALYSIS_SCHEMA = "DepPLM-Sanger-organoid-independent-validation-v2"
COMPLETION_SCHEMA = "DepPLM-Sanger-organoid-completion-v2"
VERIFICATION_SCHEMA = "DepPLM-Sanger-organoid-verification-v2"
TRAINING_RUNTIME_SCHEMA = "DepPLM-Sanger-organoid-training-runtime-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
CUDA_DRIVER_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")

TRAINING_RUNTIME_FIELDS = frozenset(
    {
        "schema",
        "python_version",
        "python_implementation",
        "platform",
        "machine",
        "package_versions",
        "torch_module_version",
        "device_type",
        "cuda_available",
        "cuda_runtime_version",
        "cuda_driver_version",
        "cuda_device_index",
        "cuda_device_name",
        "cuda_device_capability",
        "deterministic_algorithms_enabled",
        "deterministic_debug_mode",
        "cudnn_version",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cublas_workspace_config",
    }
)

_TRUSTED_LOCAL_MODULES: dict[str, tuple[Any, dict[str, Any], str]] = {}
_TRUSTED_RUNTIME_MODULES: dict[str, Any] = {}
_RUNTIME_EXECUTABLE_NAMES = {"pyreadr": ("read_r",)}
_TRUSTED_RUNTIME_EXECUTABLES: dict[str, dict[str, tuple[Any, str, dict[str, Any]]]] = {}
_PROCESS_GLOBAL_STATE_LOCK = threading.RLock()
_RESOURCE_CLOSE_LOCK = threading.RLock()
_LOCK_PREDICTIONS_OWNER_THREAD_ID: int | None = None


def _formal_invocation_constraint() -> dict[str, Any]:
    """Return the process-creation constraint bound into every protocol lock."""
    return {
        "command": FORMAL_LAUNCH_COMMAND,
        "isolated_python_required": True,
        "isolated_flag": "-I",
        "launcher": _LAUNCHER_SOURCE_RELATIVE,
    }


class _TrustedRuntimeProxy:
    """Resolve a scientific module only through its installed distribution."""

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name

    def __getattr__(self, name: str) -> Any:
        module = _load_trusted_runtime_module(self._module_name)
        return getattr(module, name)


# Postponed annotations keep these proxies inert during controller execution.
# The analyze/verify entry points establish externally supplied marker trust before
# any attribute access can execute third-party package code.
np = _TrustedRuntimeProxy("numpy")
pd = _TrustedRuntimeProxy("pandas")
stats = _TrustedRuntimeProxy("scipy.stats")
pyreadr = _TrustedRuntimeProxy("pyreadr")

TRAINING_INPUTS = (
    "data/processed/genes.txt",
    "data/processed/cell_lines.txt",
    "data/processed/lineages.txt",
    "data/processed/gene_effect.npy",
    "data/processed/ppi_graph.npz",
    "data/processed/gene_seqs.fa",
    "data/processed/esm2_t6_8M_residue_chunked.npy",
    "data/processed/esm2_t6_8M_residue_chunked.l2.npy",
    "data/processed/esm2_t6_8M_residue_chunked.names.txt",
    "data/processed/esm2_t6_8M_residue_chunked.manifest.json",
    "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "data/depmap/OmicsSomaticMutations.csv",
)
CODE_DEPENDENCIES = (
    "code/organoid_independent_validation.py",
    "code/run_organoid_independent_validation.py",
    "code/run_corrected_experiments.py",
    "code/corrected_protocol.py",
    "code/models.py",
    "code/03_embed_proteins.py",
    "code/embedding_provenance.py",
    "code/sanger_independent_validation.py",
    "docs/revision_v2/ORGANOID_INDEPENDENT_VALIDATION_PROTOCOL.md",
)
TRAINING_LOCAL_MODULES = (
    "corrected_protocol",
    "embedding_provenance",
    "models",
    "run_corrected_experiments",
)

HISTORICAL_REVISION_PROVENANCE = {
    "status": "post-download fixed-protocol revision after a pre-numeric-access workflow failure; not a prospective preregistration",
    "original_protocol_lock": {
        "path": "results/organoid_independent_validation_v1/protocol-lock.json",
        "created_utc_in_json": "2026-10-22T12:54:43Z",
        "filesystem_last_write_utc_observed": "2026-10-22T12:54:44.4210103Z",
        "bytes": 44_379,
        "sha256": "c4d270fa4698c50cf30ed19454ec3b1357dcb128e38dd3a984a7b81523ab6ac4",
        "lock_body_sha256": "05de8715e2547d67f84ff315b1a68793c28329bc5061a363cb78467bf302185f",
    },
    "locally_retained_freeze_receipt": {
        "recorded_at_utc": "2026-10-22T12:55:00.2031740Z",
        "sha256": "c4d270fa4698c50cf30ed19454ec3b1357dcb128e38dd3a984a7b81523ab6ac4",
        "limitation": "ordinary local storage; not WORM, signed, public, or DOI-backed",
    },
    "outcome_download_receipt": {
        "download_completed_at_utc": "2026-10-22T13:04:26.4301270Z",
        "local_sha256": "0d2f7d3300affee717d87e621a74c90ed6917d8491ae048e4720a89531aabb07",
        "preceding_protocol_lock_sha256": "c4d270fa4698c50cf30ed19454ec3b1357dcb128e38dd3a984a7b81523ab6ac4",
        "limitation": "ordinary local storage; not WORM, signed, public, or DOI-backed",
    },
    "failed_attempt": {
        "directory": "results/organoid_independent_validation_v1/20261022T145510.654598Z-organoid-bf6f603c57aa",
        "filesystem_creation_utc_observed": "2026-10-22T14:55:10.6557540Z",
        "filesystem_last_write_utc_observed": "2026-10-22T14:55:44.9041159Z",
        "observed_error": "Broad training outcomes contain NaN or infinite values",
        "artifacts_written": 0,
        "workflow_scope_statement": "this failed workflow execution wrote no prediction lock or outcome snapshot and did not copy or parse the numeric outcome; this does not establish what a person or another process accessed",
    },
}


class ValidationError(ValueError):
    """Raised when a fail-closed integrity or scientific gate fails."""


CleanupFailure = tuple[str, BaseException]


def _capture_cleanup_failure(
    failures: list[CleanupFailure],
    label: str,
    cleanup: Callable[[], None],
) -> None:
    """Run one cleanup step without allowing it to hide an earlier failure."""
    try:
        cleanup()
    except BaseException as exc:
        failures.append((label, exc))


def _raise_after_cleanup(
    primary: BaseException | None,
    cleanup_failures: Sequence[CleanupFailure],
) -> NoReturn:
    """Propagate the primary failure while retaining cleanup diagnostics."""
    if primary is None:
        if not cleanup_failures:
            raise RuntimeError("failure propagation requested without an exception")
        primary_label, primary = cleanup_failures[0]
        primary.add_note(f"cleanup step failed: {primary_label}")
        diagnostic_failures = list(cleanup_failures[1:])
    else:
        diagnostic_failures = list(cleanup_failures)

    for label, failure in diagnostic_failures:
        primary.add_note(
            f"additional cleanup failure [{label}]: "
            f"{type(failure).__name__}: {failure}"
        )
    distinct_diagnostics = [
        failure for _, failure in diagnostic_failures if failure is not primary
    ]
    if not distinct_diagnostics:
        raise primary.with_traceback(primary.__traceback__)
    if len(distinct_diagnostics) == 1:
        diagnostic: BaseException = distinct_diagnostics[0]
    else:
        diagnostic = BaseExceptionGroup(
            "cleanup failures raised while preserving the primary failure",
            distinct_diagnostics,
        )
    raise primary.with_traceback(primary.__traceback__) from diagnostic


DescriptorIdentity = tuple[int, int, int]


def _descriptor_identity(observed: os.stat_result) -> DescriptorIdentity:
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(stat_module.S_IFMT(observed.st_mode)),
    )


class _NativeDescriptorToken(ctypes.c_int):
    """A native-return descriptor whose Python object owns the fd immediately."""

    def fileno(self) -> int:
        descriptor = int(self.value)
        if descriptor < 0:
            raise ValidationError("native descriptor ownership has been retired")
        return descriptor

    def relinquish(self) -> None:
        self.value = -1

    def close(self) -> None:
        descriptor = int(self.value)
        if descriptor < 0:
            return
        # A close failure is ambiguous: the descriptor may already have been
        # consumed and its numeric value may be reused before Python regains
        # control.  Retire the token before entering the OS call so neither an
        # owner nor ``__del__`` can ever retry that value.
        self.value = -1
        os.close(descriptor)

    def close_quietly(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def __del__(self) -> None:
        self.close_quietly()


def _native_descriptor_result(function: Any, *arguments: Any) -> _NativeDescriptorToken:
    """Call a native fd-returning function with RAII installed by ctypes."""
    function.restype = _NativeDescriptorToken
    token = function(*arguments)
    if token.value < 0:
        error = ctypes.get_errno() or errno.EIO
        token.relinquish()
        raise OSError(error, os.strerror(error))
    return token


def _native_open_descriptor(
    path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    flags: int,
    mode: int = 0o777,
    *,
    dir_fd: int | None = None,
) -> _NativeDescriptorToken:
    """Open a POSIX fd whose native return value is already an owning object."""
    if os.name == "nt":
        raise ValidationError("native POSIX descriptor opening is unavailable")
    encoded_path = os.fsencode(path)
    if b"\0" in encoded_path:
        raise ValueError("embedded null byte")
    libc = ctypes.CDLL(None, use_errno=True)
    openat = getattr(libc, "openat", None)
    if openat is None:
        raise ValidationError("native openat descriptor acquisition is unavailable")
    openat.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint)
    return _native_descriptor_result(
        openat,
        -100 if dir_fd is None else dir_fd,
        encoded_path,
        flags,
        mode,
    )


def _native_duplicate_descriptor(descriptor: int) -> _NativeDescriptorToken:
    """Duplicate a descriptor with RAII attached before Python sees the result."""
    library_name: str | None = "ucrtbase" if os.name == "nt" else None
    runtime = ctypes.CDLL(library_name, use_errno=True)
    duplicate = runtime._dup if os.name == "nt" else runtime.dup
    duplicate.argtypes = (ctypes.c_int,)
    return _native_descriptor_result(duplicate, descriptor)


def _native_open_osfhandle(handle: int, flags: int) -> _NativeDescriptorToken:
    """Transfer a Windows HANDLE into a RAII CRT descriptor result."""
    if os.name != "nt":
        raise ValidationError("Windows HANDLE conversion is unavailable")
    runtime = ctypes.CDLL("ucrtbase", use_errno=True)
    open_osfhandle = runtime._open_osfhandle
    open_osfhandle.argtypes = (ctypes.c_ssize_t, ctypes.c_int)
    return _native_descriptor_result(open_osfhandle, handle, flags)


def _linux_kcmp_number() -> int:
    machine = platform.machine().lower()
    numbers = {
        "x86_64": 312,
        "amd64": 312,
        "i386": 349,
        "i686": 349,
        "aarch64": 272,
        "arm64": 272,
        "armv7l": 378,
        "ppc64": 312,
        "ppc64le": 312,
        "s390x": 343,
        "riscv64": 272,
    }
    try:
        return numbers[machine]
    except KeyError as exc:
        raise OSError(
            errno.ENOTSUP, "kcmp is unavailable on this architecture"
        ) from exc


def _same_open_descriptor(candidate: int, anchor: int) -> bool:
    """Compare open instances without mutating either descriptor's state."""
    if os.name == "nt":
        import msvcrt

        candidate_handle = msvcrt.get_osfhandle(candidate)
        anchor_handle = msvcrt.get_osfhandle(anchor)
        compare = ctypes.WinDLL("KernelBase", use_last_error=True).CompareObjectHandles
        compare.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        compare.restype = ctypes.c_bool
        return bool(compare(candidate_handle, anchor_handle))
    if not sys.platform.startswith("linux"):
        raise OSError(errno.ENOTSUP, "open-file-description comparison is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.argtypes = (
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    syscall.restype = ctypes.c_long
    result = syscall(
        _linux_kcmp_number(),
        os.getpid(),
        os.getpid(),
        0,  # KCMP_FILE
        candidate,
        anchor,
    )
    if result < 0:
        error = ctypes.get_errno() or errno.EIO
        raise OSError(error, os.strerror(error))
    return result == 0


class _OwnedDescriptor:
    """Own one descriptor without ever retrying an ambiguous close."""

    __slots__ = ("_token", "_anchor", "_identity")

    def __init__(self, descriptor: int | None = None) -> None:
        self._token: _NativeDescriptorToken | None = None
        self._anchor: _NativeDescriptorToken | None = None
        self._identity: DescriptorIdentity | None = None
        if descriptor is not None:
            self._adopt(_NativeDescriptorToken(descriptor))

    @classmethod
    def acquire(
        cls,
        token: _NativeDescriptorToken,
    ) -> _OwnedDescriptor:
        """Bind a descriptor token that owned the native result at return time."""
        owner = cls()
        primary_failure: BaseException | None = None
        try:
            owner._token = token
            owner.bind_current_identity()
            return owner
        except BaseException as exc:
            primary_failure = exc

        cleanup_failures: list[CleanupFailure] = []
        if owner._token is token:
            _capture_cleanup_failure(
                cleanup_failures,
                "closing descriptor after acquisition failure",
                owner.close,
            )
        else:
            _capture_cleanup_failure(
                cleanup_failures,
                "closing unregistered descriptor token",
                token.close,
            )
        _raise_after_cleanup(primary_failure, cleanup_failures)

    @classmethod
    def open(
        cls,
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
        acquisition_oserror: Callable[[OSError], BaseException] | None = None,
    ) -> _OwnedDescriptor:
        try:
            token = _native_open_descriptor(path, flags, mode, dir_fd=dir_fd)
        except OSError as exc:
            if acquisition_oserror is None:
                raise
            translated = acquisition_oserror(exc)
            if translated is exc:
                raise
            raise translated from exc
        return cls.acquire(token)

    @classmethod
    def duplicate(cls, descriptor: int) -> _OwnedDescriptor:
        return cls.acquire(_native_duplicate_descriptor(descriptor))

    def _adopt(self, token: _NativeDescriptorToken) -> None:
        self._token = token
        primary_failure: BaseException | None = None
        try:
            self.bind_current_identity()
            return
        except BaseException as exc:
            primary_failure = exc

        cleanup_failures: list[CleanupFailure] = []
        _capture_cleanup_failure(
            cleanup_failures,
            "closing descriptor after ownership binding failure",
            self.close,
        )
        _raise_after_cleanup(primary_failure, cleanup_failures)

    def fileno(self) -> int:
        token = self._token
        if token is None:
            raise ValidationError("descriptor ownership has already been retired")
        return token.fileno()

    def bind_current_identity(self) -> os.stat_result:
        observed = os.fstat(self.fileno())
        self._identity = _descriptor_identity(observed)
        anchor = _native_duplicate_descriptor(self.fileno())
        self._anchor = anchor
        if not _same_open_descriptor(self.fileno(), anchor.fileno()):
            raise ValidationError("descriptor anchor is not the same open instance")
        return observed

    def close(self) -> None:
        cleanup_failures: list[CleanupFailure] = []
        primary_failure: BaseException | None = None
        with _RESOURCE_CLOSE_LOCK:
            token = self._token
            anchor = self._anchor
            if token is None:
                self._anchor = None
                self._identity = None
                if anchor is not None:
                    _capture_cleanup_failure(
                        cleanup_failures,
                        "closing detached descriptor identity anchor",
                        anchor.close,
                    )
                if cleanup_failures:
                    _raise_after_cleanup(None, cleanup_failures)
                return
            if int(token.value) < 0:
                self._token = None
                self._anchor = None
                self._identity = None
                if anchor is not None:
                    _capture_cleanup_failure(
                        cleanup_failures,
                        "closing descriptor anchor after token retirement",
                        anchor.close,
                    )
                if cleanup_failures:
                    _raise_after_cleanup(None, cleanup_failures)
                return
            descriptor = token.fileno()
            should_close_candidate = False
            if self._identity is None and anchor is None:
                # Binding failed before any identity observation.  The native
                # return token is still the only owner and must be retired.
                should_close_candidate = True
            else:
                try:
                    try:
                        current = os.fstat(descriptor)
                    except OSError as exc:
                        if exc.errno == errno.EBADF:
                            token.relinquish()
                            self._token = None
                            self._anchor = None
                            self._identity = None
                            if anchor is not None:
                                _capture_cleanup_failure(
                                    cleanup_failures,
                                    "closing descriptor identity anchor",
                                    anchor.close,
                                )
                            if cleanup_failures:
                                _raise_after_cleanup(None, cleanup_failures)
                            return
                        raise
                    if self._identity is not None and (
                        _descriptor_identity(current) != self._identity
                    ):
                        raise ValidationError(
                            "owned descriptor identity changed; refusing to close a "
                            "possibly reused descriptor"
                        )
                    if anchor is not None and not _same_open_descriptor(
                        descriptor, anchor.fileno()
                    ):
                        raise ValidationError(
                            "owned descriptor open instance changed; refusing to close a "
                            "reused descriptor"
                        )
                    should_close_candidate = True
                except BaseException as exc:
                    primary_failure = exc

            if should_close_candidate:
                # Detach both tokens before either close attempt.  Each native
                # token enters a permanent non-retry state before its OS call.
                self._token = None
                self._anchor = None
                self._identity = None
                _capture_cleanup_failure(
                    cleanup_failures,
                    "closing owned descriptor",
                    token.close,
                )
            else:
                token.relinquish()
                self._token = None
                self._anchor = None
                self._identity = None
            if anchor is not None:
                _capture_cleanup_failure(
                    cleanup_failures,
                    "closing descriptor identity anchor",
                    anchor.close,
                )
        if primary_failure is not None or cleanup_failures:
            _raise_after_cleanup(primary_failure, cleanup_failures)

    def close_quietly(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def __enter__(self) -> _OwnedDescriptor:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        primary: BaseException | None,
        _traceback: Any,
    ) -> None:
        cleanup_failures: list[CleanupFailure] = []
        _capture_cleanup_failure(
            cleanup_failures,
            "closing owned descriptor",
            self.close,
        )
        if primary is not None or cleanup_failures:
            _raise_after_cleanup(primary, cleanup_failures)

    def __del__(self) -> None:
        self.close_quietly()


class _DescriptorBackedStream:
    """A non-owning ``fdopen`` view paired with the descriptor's owner."""

    __slots__ = ("_stream", "_owner")

    def __init__(self, stream: Any, owner: _OwnedDescriptor) -> None:
        self._stream = stream
        self._owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __enter__(self) -> _DescriptorBackedStream:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        primary: BaseException | None,
        _traceback: Any,
    ) -> None:
        cleanup_failures: list[CleanupFailure] = []
        _capture_cleanup_failure(
            cleanup_failures,
            "closing non-owning descriptor stream",
            self._stream.close,
        )
        _capture_cleanup_failure(
            cleanup_failures,
            "closing descriptor backing stream",
            self._owner.close,
        )
        if primary is not None or cleanup_failures:
            _raise_after_cleanup(primary, cleanup_failures)

    def close(self) -> None:
        cleanup_failures: list[CleanupFailure] = []
        _capture_cleanup_failure(
            cleanup_failures,
            "closing non-owning descriptor stream",
            self._stream.close,
        )
        _capture_cleanup_failure(
            cleanup_failures,
            "closing descriptor backing stream",
            self._owner.close,
        )
        if cleanup_failures:
            _raise_after_cleanup(None, cleanup_failures)

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


def _open_owned_descriptor_stream(
    owner: _OwnedDescriptor, mode: str
) -> _DescriptorBackedStream:
    """Create an fdopen view while retaining explicit descriptor ownership."""
    primary_failure: BaseException | None = None
    stream: Any | None = None
    try:
        if owner._identity is None:
            owner.bind_current_identity()
        descriptor = owner.fileno()
        stream = os.fdopen(descriptor, mode, closefd=False)
        if stream.fileno() != descriptor:
            raise ValidationError("fdopen returned a view of an unexpected descriptor")
        return _DescriptorBackedStream(stream, owner)
    except BaseException as exc:
        primary_failure = exc

    cleanup_failures: list[CleanupFailure] = []
    if stream is not None:
        _capture_cleanup_failure(
            cleanup_failures,
            "closing failed non-owning descriptor stream",
            stream.close,
        )
    _capture_cleanup_failure(
        cleanup_failures,
        "closing descriptor after fdopen failure",
        owner.close,
    )
    _raise_after_cleanup(primary_failure, cleanup_failures)


def _duplicate_descriptor_stream(descriptor: int, mode: str) -> _DescriptorBackedStream:
    """Duplicate *descriptor* and immediately put the copy under ownership."""
    owner = _OwnedDescriptor.duplicate(descriptor)
    return _open_owned_descriptor_stream(owner, mode)


def _register_lock_predictions_owner(thread_id: int) -> None:
    global _LOCK_PREDICTIONS_OWNER_THREAD_ID

    _LOCK_PREDICTIONS_OWNER_THREAD_ID = thread_id


def _restore_lock_predictions_owner(previous_owner: int | None) -> None:
    global _LOCK_PREDICTIONS_OWNER_THREAD_ID

    _LOCK_PREDICTIONS_OWNER_THREAD_ID = previous_owner


@contextmanager
def _lock_predictions_single_flight() -> Iterator[None]:
    """Serialize the complete stateful prediction-lock phase in this process."""
    global _LOCK_PREDICTIONS_OWNER_THREAD_ID

    thread_id = threading.get_ident()
    with _PROCESS_GLOBAL_STATE_LOCK:
        previous_owner = _LOCK_PREDICTIONS_OWNER_THREAD_ID
        primary_failure: BaseException | None = None
        cleanup_failures: list[CleanupFailure] = []
        try:
            if previous_owner is not None:
                if previous_owner == thread_id:
                    raise ValidationError(
                        "reentrant lock_predictions invocation is not permitted"
                    )
                raise RuntimeError(
                    "prediction-lock single-flight ownership is inconsistent"
                )
            _register_lock_predictions_owner(thread_id)
            yield
        except BaseException as exc:
            primary_failure = exc
        _capture_cleanup_failure(
            cleanup_failures,
            "initial prediction-lock owner restoration",
            lambda: _restore_lock_predictions_owner(previous_owner),
        )
        _capture_cleanup_failure(
            cleanup_failures,
            "second-chance prediction-lock owner restoration",
            lambda: _restore_lock_predictions_owner(previous_owner),
        )
        if primary_failure is not None or cleanup_failures:
            _raise_after_cleanup(primary_failure, cleanup_failures)


def _require_lock_predictions_single_flight() -> None:
    if _LOCK_PREDICTIONS_OWNER_THREAD_ID != threading.get_ident():
        raise ValidationError(
            "train_and_lock_predictions must run inside lock_predictions single-flight"
        )


@dataclass(frozen=True)
class OutcomeMatrix:
    sample_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    values: np.ndarray
    libraries_aggregated: int
    excluded_rows: int


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


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
    ).encode("ascii")


def _parse_json_mapping(payload_bytes: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValidationError(f"{label} contains duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except ValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"{label} must be a JSON object")
    return payload


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _stable_read_bytes(
    path: Path,
    *,
    label: str,
    root: Path | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Read one stable regular-file instance and bind exactly those bytes."""
    try:
        resolved = path.resolve(strict=True)
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(before.st_mode):
                raise ValidationError(f"{label} is not a regular file")
            payload = handle.read()
            after = os.fstat(handle.fileno())
        current = resolved.stat()
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(f"{label} cannot be read: {path}") from exc
    if (
        _stat_signature(before) != _stat_signature(after)
        or _stat_signature(after) != _stat_signature(current)
        or len(payload) != int(after.st_size)
    ):
        raise ValidationError(f"{label} changed while it was being read")
    record_path = (
        _portable_relative(resolved, root) if root is not None else str(resolved)
    )
    return payload, {
        "path": record_path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _load_json_mapping(
    path: Path,
    label: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    payload, _ = _stable_read_bytes(path, label=label, root=root)
    return _parse_json_mapping(payload, label)


def _load_json_mapping_with_record(
    path: Path,
    label: str,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, record = _stable_read_bytes(path, label=label, root=root)
    return _parse_json_mapping(payload, label), record


def _record_fields(record: Mapping[str, Any], label: str) -> dict[str, Any]:
    required = {"path", "bytes", "sha256"}
    if not isinstance(record, Mapping) or not required.issubset(record):
        raise ValidationError(f"{label} file record is incomplete")
    return {name: record[name] for name in ("path", "bytes", "sha256")}


def _assert_same_record(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if _record_fields(observed, label) != _record_fields(expected, label):
        raise ValidationError(f"{label} changed after binding")


def _assert_same_content(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    observed_fields = _record_fields(observed, label)
    expected_fields = _record_fields(expected, label)
    if (
        observed_fields["bytes"] != expected_fields["bytes"]
        or observed_fields["sha256"] != expected_fields["sha256"]
    ):
        raise ValidationError(f"{label} content differs from its campaign binding")


def _stable_hash_and_size(
    path: Path,
    algorithm: str,
    chunk_size: int = 4 * 1024 * 1024,
) -> tuple[str, int]:
    try:
        resolved = path.resolve(strict=True)
        digest = (
            hashlib.new(algorithm, usedforsecurity=False)
            if algorithm == "md5"
            else hashlib.new(algorithm)
        )
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(before.st_mode):
                raise ValidationError(
                    f"bound artifact is not a regular file: {resolved}"
                )
            size = 0
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(handle.fileno())
        current = resolved.stat()
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(f"required file cannot be hashed: {path}") from exc
    if (
        _stat_signature(before) != _stat_signature(after)
        or _stat_signature(after) != _stat_signature(current)
        or size != int(after.st_size)
    ):
        raise ValidationError(f"file changed while it was being hashed: {resolved}")
    return digest.hexdigest(), size


def _hash_open_handle(
    handle: Any, chunk_size: int = 4 * 1024 * 1024
) -> tuple[str, str, int]:
    """Hash one already-open file instance and rewind it for its next consumer."""
    sha256_digest = hashlib.sha256()
    md5_digest = hashlib.md5(usedforsecurity=False)
    size = 0
    try:
        handle.seek(0)
        while chunk := handle.read(chunk_size):
            sha256_digest.update(chunk)
            md5_digest.update(chunk)
            size += len(chunk)
        handle.seek(0)
    except (OSError, ValueError) as exc:
        raise ValidationError(
            "open file instance cannot be hashed and rewound"
        ) from exc
    return sha256_digest.hexdigest(), md5_digest.hexdigest(), size


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest, _ = _stable_hash_and_size(path, "sha256", chunk_size)
    return digest


def _md5_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest, _ = _stable_hash_and_size(path, "md5", chunk_size)
    return digest


def _portable_relative(path: Path, root: Path = ROOT) -> str:
    resolved = path.resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValidationError(f"path is outside the project root: {resolved}") from exc
    return PurePosixPath(relative).as_posix()


def _project_output_path(path: Path, *, root: Path = ROOT) -> Path:
    root_resolved = Path(os.path.abspath(root))
    resolved = Path(os.path.abspath(path))
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValidationError(
            f"output path is outside the project root: {resolved}"
        ) from exc
    if not relative.parts:
        raise ValidationError("output path cannot be the project root")
    return resolved


def _windows_close_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    if not close_handle(wintypes.HANDLE(handle)):
        error = ctypes.get_last_error()
        raise OSError(error, ctypes.FormatError(error))


WindowsHandleIdentity = tuple[int, int]


class _NativeWindowsHandleToken(ctypes.c_void_p):
    """A CreateFileW/DuplicateHandle result that owns its HANDLE immediately."""

    def fileno(self) -> int:
        handle = self.value
        invalid = ctypes.c_void_p(-1).value
        if handle is None or handle == invalid:
            raise ValidationError("native Windows handle ownership has been retired")
        return int(handle)

    def relinquish(self) -> None:
        self.value = None

    def close(self) -> None:
        handle = self.value
        invalid = ctypes.c_void_p(-1).value
        if handle is None or handle == invalid:
            return
        # CloseHandle has the same ambiguous-return problem as close(2): once
        # attempted, this numeric value must never be retried by finalization.
        self.value = None
        _windows_close_handle(int(handle))

    def close_quietly(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def __del__(self) -> None:
        self.close_quietly()


def _duplicate_windows_handle(handle: int) -> _NativeWindowsHandleToken:
    """Create an owning duplicate using a pre-existing output token."""
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    current_process = kernel32.GetCurrentProcess
    current_process.argtypes = ()
    current_process.restype = wintypes.HANDLE
    duplicate_handle = kernel32.DuplicateHandle
    duplicate_handle.argtypes = (
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(_NativeWindowsHandleToken),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    duplicate_handle.restype = wintypes.BOOL
    token = _NativeWindowsHandleToken()
    process = current_process()
    if not duplicate_handle(
        process,
        wintypes.HANDLE(handle),
        process,
        ctypes.byref(token),
        0,
        False,
        0x00000002,  # DUPLICATE_SAME_ACCESS
    ):
        error = ctypes.get_last_error()
        token.close_quietly()
        raise OSError(error, ctypes.FormatError(error))
    return token


def _same_windows_object(candidate: int, anchor: int) -> bool:
    compare = ctypes.WinDLL("KernelBase", use_last_error=True).CompareObjectHandles
    compare.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    compare.restype = ctypes.c_bool
    return bool(compare(candidate, anchor))


class _WindowsHandleAcquisitionError(OSError):
    """A CreateFileW failure, distinct from post-acquisition integrity errors."""

    def __init__(self, error: int, path: Path) -> None:
        import ctypes

        super().__init__(error, ctypes.FormatError(error), str(path))
        self.winerror = error


def _windows_handle_information(
    handle: int, path: Path | None = None
) -> tuple[int, WindowsHandleIdentity]:
    import ctypes
    from ctypes import wintypes

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = (
            ("attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("access_time", wintypes.FILETIME),
            ("write_time", wintypes.FILETIME),
            ("volume_serial", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        )

    information = _ByHandleFileInformation()
    get_information = ctypes.WinDLL(
        "kernel32", use_last_error=True
    ).GetFileInformationByHandle
    get_information.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    )
    get_information.restype = wintypes.BOOL
    if not get_information(wintypes.HANDLE(handle), ctypes.byref(information)):
        error = ctypes.get_last_error()
        raise OSError(
            error, ctypes.FormatError(error), None if path is None else str(path)
        )
    identity = (
        int(information.volume_serial),
        (int(information.file_index_high) << 32) | int(information.file_index_low),
    )
    return int(information.attributes), identity


class _OwnedWindowsHandle:
    """Own one Windows handle and refuse to close a reused numeric handle."""

    __slots__ = ("_token", "_anchor", "_identity")

    def __init__(self) -> None:
        self._token: _NativeWindowsHandleToken | None = None
        self._anchor: _NativeWindowsHandleToken | None = None
        self._identity: WindowsHandleIdentity | None = None

    def fileno(self) -> int:
        token = self._token
        if token is None:
            raise ValidationError("Windows handle ownership has already been retired")
        return token.fileno()

    @property
    def identity(self) -> WindowsHandleIdentity:
        if self._identity is None:
            raise ValidationError("Windows handle identity has not been bound")
        return self._identity

    def relinquish(self) -> None:
        cleanup_failures: list[CleanupFailure] = []
        with _RESOURCE_CLOSE_LOCK:
            token = self._token
            anchor = self._anchor
            if token is not None:
                # The HANDLE may already have been consumed by
                # _open_osfhandle.  Make that transfer irrevocable before any
                # anchor cleanup that can raise.
                token.relinquish()
            self._anchor = None
            self._token = None
            self._identity = None
            if anchor is not None:
                _capture_cleanup_failure(
                    cleanup_failures,
                    "closing relinquished Windows handle identity anchor",
                    anchor.close,
                )
        if cleanup_failures:
            _raise_after_cleanup(None, cleanup_failures)

    def close(self) -> None:
        cleanup_failures: list[CleanupFailure] = []
        primary_failure: BaseException | None = None
        with _RESOURCE_CLOSE_LOCK:
            token = self._token
            anchor = self._anchor
            if token is None:
                self._anchor = None
                self._identity = None
                if anchor is not None:
                    _capture_cleanup_failure(
                        cleanup_failures,
                        "closing detached Windows handle identity anchor",
                        anchor.close,
                    )
                if cleanup_failures:
                    _raise_after_cleanup(None, cleanup_failures)
                return
            invalid = ctypes.c_void_p(-1).value
            if token.value is None or token.value == invalid:
                self._token = None
                self._anchor = None
                self._identity = None
                if anchor is not None:
                    _capture_cleanup_failure(
                        cleanup_failures,
                        "closing Windows anchor after token retirement",
                        anchor.close,
                    )
                if cleanup_failures:
                    _raise_after_cleanup(None, cleanup_failures)
                return
            handle = token.fileno()
            should_close_candidate = False
            if self._identity is None and anchor is None:
                # The CreateFileW token owns the native return even when
                # identity binding is interrupted immediately afterwards.
                should_close_candidate = True
            else:
                try:
                    _, current_identity = _windows_handle_information(handle)
                except OSError as exc:
                    if getattr(exc, "winerror", None) == 6 or exc.errno == 6:
                        token.relinquish()
                        self._token = None
                        self._anchor = None
                        self._identity = None
                        if anchor is not None:
                            _capture_cleanup_failure(
                                cleanup_failures,
                                "closing Windows handle identity anchor",
                                anchor.close,
                            )
                        if cleanup_failures:
                            _raise_after_cleanup(None, cleanup_failures)
                        return
                    primary_failure = exc
                else:
                    try:
                        if (
                            self._identity is not None
                            and current_identity != self._identity
                        ):
                            raise ValidationError(
                                "owned Windows handle identity changed; refusing to "
                                "close a possibly reused handle"
                            )
                        if anchor is not None and not _same_windows_object(
                            handle, anchor.fileno()
                        ):
                            raise ValidationError(
                                "owned Windows handle open instance changed; refusing "
                                "to close a reused handle"
                            )
                        should_close_candidate = True
                    except BaseException as exc:
                        primary_failure = exc

            if should_close_candidate:
                self._token = None
                self._anchor = None
                self._identity = None
                _capture_cleanup_failure(
                    cleanup_failures,
                    "closing owned Windows handle",
                    token.close,
                )
            else:
                token.relinquish()
                self._token = None
                self._anchor = None
                self._identity = None
            if anchor is not None:
                _capture_cleanup_failure(
                    cleanup_failures,
                    "closing Windows handle identity anchor",
                    anchor.close,
                )
        if primary_failure is not None or cleanup_failures:
            _raise_after_cleanup(primary_failure, cleanup_failures)

    def close_quietly(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def __enter__(self) -> _OwnedWindowsHandle:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        primary: BaseException | None,
        _traceback: Any,
    ) -> None:
        cleanup_failures: list[CleanupFailure] = []
        _capture_cleanup_failure(
            cleanup_failures,
            "closing owned Windows handle",
            self.close,
        )
        if primary is not None or cleanup_failures:
            _raise_after_cleanup(primary, cleanup_failures)

    def __del__(self) -> None:
        self.close_quietly()


def _acquire_windows_handle(
    acquisition: Callable[[], Any], path: Path
) -> tuple[_OwnedWindowsHandle, int]:
    """Bind a CreateFileW result that is already protected by a native token."""
    owner = _OwnedWindowsHandle()
    primary_failure: BaseException | None = None
    try:
        token = acquisition()
        if not isinstance(token, _NativeWindowsHandleToken):
            raise TypeError("Windows acquisition did not return an owning token")
        if token.value == ctypes.c_void_p(-1).value:
            error = ctypes.get_last_error()
            token.relinquish()
            raise _WindowsHandleAcquisitionError(error, path)
        owner._token = token
        attributes, owner._identity = _windows_handle_information(owner.fileno(), path)
        anchor = _duplicate_windows_handle(owner.fileno())
        owner._anchor = anchor
        if not _same_windows_object(owner.fileno(), anchor.fileno()):
            raise ValidationError("Windows handle anchor is not the same open instance")
        return owner, attributes
    except BaseException as exc:
        primary_failure = exc

    cleanup_failures: list[CleanupFailure] = []
    _capture_cleanup_failure(
        cleanup_failures,
        f"closing Windows handle after acquisition failure: {path}",
        owner.close,
    )
    _raise_after_cleanup(primary_failure, cleanup_failures)


def _windows_open_bound_handle(path: Path, *, directory: bool) -> _OwnedWindowsHandle:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = _NativeWindowsHandleToken
    owner, attributes = _acquire_windows_handle(
        lambda: create_file(
            str(path),
            0x0080,  # FILE_READ_ATTRIBUTES
            (
                0x00000001 | 0x00000002 if directory else 0x00000001
            ),  # never share delete; source files are also write-excluded
            None,
            3,  # OPEN_EXISTING
            0x00200000 | (0x02000000 if directory else 0),
            None,
        ),
        path,
    )
    primary_failure: BaseException | None = None
    try:
        if attributes & 0x00000400:  # FILE_ATTRIBUTE_REPARSE_POINT
            raise ValidationError(f"output path traverses a reparse point: {path}")
        is_directory = bool(attributes & 0x00000010)
        if is_directory != directory:
            kind = "directory" if directory else "regular file"
            raise ValidationError(f"bound path is not a {kind}: {path}")
        return owner
    except BaseException as exc:
        primary_failure = exc

    cleanup_failures: list[CleanupFailure] = []
    _capture_cleanup_failure(
        cleanup_failures,
        f"closing rejected bound Windows handle: {path}",
        owner.close,
    )
    _raise_after_cleanup(primary_failure, cleanup_failures)


def _windows_handle_to_descriptor(
    handle_owner: _OwnedWindowsHandle, flags: int
) -> _OwnedDescriptor:
    """Transfer a Windows handle to a CRT descriptor without dual ownership."""
    descriptor_owner: _OwnedDescriptor | None = None
    primary_failure: BaseException | None = None
    try:
        descriptor_owner = _OwnedDescriptor.acquire(
            _native_open_osfhandle(handle_owner.fileno(), flags)
        )
        handle_owner.relinquish()
        return descriptor_owner
    except BaseException as exc:
        primary_failure = exc

    cleanup_failures: list[CleanupFailure] = []
    if descriptor_owner is not None:
        _capture_cleanup_failure(
            cleanup_failures,
            "closing descriptor after Windows handle transfer failure",
            descriptor_owner.close,
        )
    _capture_cleanup_failure(
        cleanup_failures,
        "closing Windows handle after descriptor transfer failure",
        handle_owner.close,
    )
    _raise_after_cleanup(primary_failure, cleanup_failures)


def _windows_create_bound_file(path: Path) -> Any:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = _NativeWindowsHandleToken
    try:
        handle_owner, _ = _acquire_windows_handle(
            lambda: create_file(
                str(path),
                0x80000000
                | 0x40000000
                | 0x0080,  # GENERIC_READ | GENERIC_WRITE | FILE_READ_ATTRIBUTES
                0x00000001,  # share read, deliberately not write/delete
                None,
                1,  # CREATE_NEW
                0x00000080 | 0x00200000,
                None,
            ),
            path,
        )
    except _WindowsHandleAcquisitionError as exc:
        if getattr(exc, "winerror", None) in {80, 183} or exc.errno in {80, 183}:
            raise FileExistsError(exc.errno, str(exc), str(path)) from exc
        raise
    descriptor_owner = _windows_handle_to_descriptor(
        handle_owner, os.O_RDWR | getattr(os, "O_BINARY", 0)
    )
    return _open_owned_descriptor_stream(descriptor_owner, "w+b")


def _posix_link_open_file(
    descriptor: int, destination_directory_descriptor: int, destination_name: str
) -> None:
    """Hard-link the inode held by *descriptor*, never a mutable source name."""
    import ctypes

    if os.name == "nt":
        raise ValidationError("POSIX descriptor linking is unavailable on Windows")
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = getattr(libc, "linkat", None)
    if linkat is None:
        raise ValidationError("descriptor-bound hard linking is unavailable")
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    encoded_destination = os.fsencode(destination_name)

    # Linux supports an empty source path for an already-open file.  Some
    # kernels/filesystems require the procfs descriptor symlink instead; that
    # fallback still names the held descriptor rather than the mutable leaf.
    attempts = ((descriptor, b"", 0x1000),)
    proc_descriptor = Path(f"/proc/self/fd/{descriptor}")
    if proc_descriptor.exists():
        attempts += ((-100, os.fsencode(proc_descriptor), 0x400),)
    dev_descriptor = Path(f"/dev/fd/{descriptor}")
    if dev_descriptor.exists() and dev_descriptor != proc_descriptor:
        attempts += ((-100, os.fsencode(dev_descriptor), 0x400),)

    last_error = errno.ENOTSUP
    for source_directory, source_name, flags in attempts:
        if (
            linkat(
                source_directory,
                source_name,
                destination_directory_descriptor,
                encoded_destination,
                flags,
            )
            == 0
        ):
            return
        last_error = ctypes.get_errno()
        if last_error == errno.EEXIST:
            raise FileExistsError(last_error, os.strerror(last_error), destination_name)
        if last_error not in {
            errno.EACCES,
            errno.EINVAL,
            errno.ENOENT,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
            errno.EPERM,
        }:
            raise OSError(last_error, os.strerror(last_error), destination_name)
    raise ValidationError(
        "the filesystem cannot securely link an already-open temporary artifact"
    ) from OSError(last_error, os.strerror(last_error), destination_name)


def _posix_rename_noreplace(
    directory_descriptor: int, source_name: str, destination_name: str
) -> None:
    """Atomically quarantine a directory entry without replacing another entry."""
    import ctypes

    if os.name == "nt":
        raise ValidationError("POSIX atomic quarantine is unavailable on Windows")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ValidationError(
            "identity-bound cleanup is unavailable; the artifact was preserved"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            directory_descriptor,
            os.fsencode(source_name),
            directory_descriptor,
            os.fsencode(destination_name),
            1,  # RENAME_NOREPLACE
        )
        == 0
    ):
        return
    error = ctypes.get_errno()
    if error == errno.ENOENT:
        raise FileNotFoundError(error, os.strerror(error), source_name)
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), destination_name)
    if error in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
        raise ValidationError(
            "identity-bound cleanup is unavailable; the artifact was preserved"
        ) from OSError(error, os.strerror(error), source_name)
    raise OSError(error, os.strerror(error), source_name)


def _windows_unlink_path_if_identity(
    path: Path, expected_identity: tuple[int, int]
) -> None:
    """Delete exactly the Windows file opened by handle, or preserve the leaf."""
    import ctypes
    from ctypes import wintypes

    class _FileDispositionInformationEx(ctypes.Structure):
        _fields_ = (("flags", wintypes.DWORD),)

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = _NativeWindowsHandleToken
    try:
        handle_owner, attributes = _acquire_windows_handle(
            lambda: create_file(
                str(path),
                0x00010000 | 0x0080,  # DELETE | FILE_READ_ATTRIBUTES
                0x00000001,  # share read, deliberately deny write/delete races
                None,
                3,  # OPEN_EXISTING
                0x00200000,  # FILE_FLAG_OPEN_REPARSE_POINT
                None,
            ),
            path,
        )
    except _WindowsHandleAcquisitionError as exc:
        error = getattr(exc, "winerror", None) or exc.errno
        if error in {2, 3}:
            return
        raise ValidationError(
            "identity-bound cleanup could not open the artifact; it was preserved"
        ) from exc
    with handle_owner:
        if attributes & 0x00000400 or attributes & 0x00000010:
            raise ValidationError(
                "identity-bound cleanup encountered a non-regular artifact; "
                "it was preserved"
            )
        try:
            observed = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise ValidationError(
                "identity-bound cleanup could not stat the held artifact; "
                "it was preserved"
            ) from exc
        if not stat_module.S_ISREG(observed.st_mode):
            raise ValidationError(
                "identity-bound cleanup encountered a non-regular artifact; "
                "it was preserved"
            )
        observed_identity = (observed.st_dev, observed.st_ino)
        if observed_identity != expected_identity:
            return
        disposition = _FileDispositionInformationEx(
            0x00000001 | 0x00000002 | 0x00000010
        )  # DELETE | POSIX_SEMANTICS | IGNORE_READONLY_ATTRIBUTE
        set_information = kernel32.SetFileInformationByHandle
        set_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        set_information.restype = wintypes.BOOL
        if not set_information(
            wintypes.HANDLE(handle_owner.fileno()),
            21,  # FileDispositionInfoEx
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            error = ctypes.get_last_error()
            raise ValidationError(
                "identity-bound cleanup could not mark the artifact for deletion; "
                "it was preserved"
            ) from OSError(error, ctypes.FormatError(error), str(path))


@dataclass
class _BoundOutputDirectory:
    path: Path
    posix_chain: list[tuple[Path, _OwnedDescriptor, tuple[int, int]]]
    windows_chain: list[tuple[Path, _OwnedWindowsHandle, tuple[int, int]]]

    @property
    def dir_fd(self) -> int:
        if not self.posix_chain:
            raise ValidationError("directory-descriptor output binding is unavailable")
        return self.posix_chain[-1][1].fileno()

    def assert_current(self) -> None:
        if os.name == "nt":
            for path, _, expected_identity in self.windows_chain:
                with _windows_open_bound_handle(path, directory=True) as current_owner:
                    if current_owner.identity != expected_identity:
                        raise ValidationError(
                            f"output directory identity changed while bound: {path}"
                        )
            return
        for path, descriptor_owner, expected_identity in self.posix_chain:
            try:
                current = os.stat(path, follow_symlinks=False)
                bound = os.fstat(descriptor_owner.fileno())
            except OSError as exc:
                raise ValidationError(
                    f"output directory changed while bound: {path}"
                ) from exc
            if (
                not stat_module.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != expected_identity
                or (bound.st_dev, bound.st_ino) != expected_identity
            ):
                raise ValidationError(
                    f"output directory identity changed while bound: {path}"
                )

    def create_file(self, name: str | None) -> Any:
        if os.name == "nt":
            if name is None:
                raise ValidationError(
                    "Windows write-once artifacts require a leaf name"
                )
            return _windows_create_bound_file(self.path / name)
        if name is not None:
            raise ValidationError(
                "POSIX write-once artifacts require an unnamed temporary inode"
            )
        temporary_flag = getattr(os, "O_TMPFILE", None)
        if not isinstance(temporary_flag, int):
            raise ValidationError(
                "secure unnamed temporary files are unavailable on this platform"
            )
        flags = os.O_RDWR | temporary_flag | getattr(os, "O_CLOEXEC", 0)
        descriptor_owner = _OwnedDescriptor.open(
            ".",
            flags,
            0o600,
            dir_fd=self.dir_fd,
            acquisition_oserror=lambda _exc: ValidationError(
                "the output filesystem cannot create a secure unnamed temporary file"
            ),
        )
        return _open_owned_descriptor_stream(descriptor_owner, "w+b")

    def file_identity(self, name: str) -> tuple[int, int]:
        if os.name == "nt":
            with _windows_open_bound_handle(self.path / name, directory=False):
                observed = os.stat(self.path / name, follow_symlinks=False)
                if not stat_module.S_ISREG(observed.st_mode):
                    raise ValidationError(
                        f"output artifact is not a regular file: {name}"
                    )
                return observed.st_dev, observed.st_ino
        observed = os.stat(name, dir_fd=self.dir_fd, follow_symlinks=False)
        if not stat_module.S_ISREG(observed.st_mode):
            raise ValidationError(f"output artifact is not a regular file: {name}")
        return observed.st_dev, observed.st_ino

    def link_open_file(
        self,
        handle: Any,
        source_name: str | None,
        destination_name: str,
        expected_identity: tuple[int, int],
    ) -> None:
        descriptor_stat = os.fstat(handle.fileno())
        descriptor_identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        if (
            not stat_module.S_ISREG(descriptor_stat.st_mode)
            or descriptor_identity != expected_identity
            or (
                source_name is not None
                and self.file_identity(source_name) != expected_identity
            )
        ):
            raise ValidationError(
                "temporary artifact identity changed before write-once commit"
            )
        if os.name == "nt":
            if source_name is None:
                raise ValidationError("Windows write-once source name is absent")
            os.link(self.path / source_name, self.path / destination_name)
        else:
            _posix_link_open_file(handle.fileno(), self.dir_fd, destination_name)
        try:
            if (
                source_name is not None
                and self.file_identity(source_name) != expected_identity
            ):
                raise ValidationError(
                    "temporary artifact name changed during descriptor-bound linking"
                )
            destination_identity = self.file_identity(destination_name)
        except BaseException as verification_failure:
            cleanup_failures: list[CleanupFailure] = []
            _capture_cleanup_failure(
                cleanup_failures,
                "removing destination after post-link verification failure",
                lambda: self.unlink_if_identity(destination_name, expected_identity),
            )
            _raise_after_cleanup(verification_failure, cleanup_failures)
        if destination_identity != expected_identity:
            identity_failure = ValidationError(
                "write-once destination is not bound to the temporary file descriptor"
            )
            cleanup_failures = []
            _capture_cleanup_failure(
                cleanup_failures,
                "removing destination with an unexpected post-link identity",
                lambda: self.unlink_if_identity(destination_name, destination_identity),
            )
            _raise_after_cleanup(identity_failure, cleanup_failures)

    def unlink_if_identity(self, name: str, expected_identity: tuple[int, int]) -> None:
        if os.name == "nt":
            _windows_unlink_path_if_identity(self.path / name, expected_identity)
            return

        quarantine_name = f".{name}.quarantine-{uuid.uuid4().hex}"
        try:
            _posix_rename_noreplace(self.dir_fd, name, quarantine_name)
        except FileNotFoundError:
            return
        try:
            quarantined_identity = self.file_identity(quarantine_name)
        except BaseException as exc:
            raise ValidationError(
                "quarantined artifact identity cannot be verified; it was preserved"
            ) from exc
        if quarantined_identity == expected_identity:
            # POSIX has no portable identity-conditional unlink.  Keeping the owned
            # inode under an unpredictable quarantine name removes the live leaf
            # without risking deletion of a concurrently substituted artifact.
            return
        try:
            _posix_rename_noreplace(self.dir_fd, quarantine_name, name)
        except BaseException as exc:
            raise ValidationError(
                "artifact changed during identity-bound cleanup; the uncertain "
                f"entry was preserved as {quarantine_name}"
            ) from exc
        raise ValidationError(
            "artifact changed during identity-bound cleanup; the replacement "
            "was restored and preserved"
        )

    def unlink(self, name: str, *, missing_ok: bool) -> None:
        try:
            if os.name == "nt":
                (self.path / name).unlink()
            else:
                os.unlink(name, dir_fd=self.dir_fd)
        except FileNotFoundError:
            if not missing_ok:
                raise


def _lexical_project_member(
    path: Path, root: Path
) -> tuple[Path, Path, tuple[str, ...]]:
    root_absolute = Path(os.path.abspath(root))
    path_absolute = Path(os.path.abspath(path))
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValidationError(
            f"output path is outside the project root: {path_absolute}"
        ) from exc
    return root_absolute, path_absolute, relative.parts


@contextmanager
def _bound_output_directory(
    directory: Path,
    *,
    root: Path,
    create: bool,
) -> Iterator[_BoundOutputDirectory]:
    """Bind a non-reparse directory chain for the complete write window."""
    root_absolute, directory_absolute, parts = _lexical_project_member(directory, root)
    posix_chain: list[tuple[Path, _OwnedDescriptor, tuple[int, int]]] = []
    windows_chain: list[tuple[Path, _OwnedWindowsHandle, tuple[int, int]]] = []
    pending_posix_owner: _OwnedDescriptor | None = None
    pending_windows_owner: _OwnedWindowsHandle | None = None
    primary_failure: BaseException | None = None
    cleanup_failures: list[CleanupFailure] = []
    try:
        if os.name == "nt":
            current = root_absolute
            try:
                pending_windows_owner = _windows_open_bound_handle(
                    current, directory=True
                )
            except _WindowsHandleAcquisitionError as exc:
                raise ValidationError(
                    f"project root cannot be bound for output: {current}"
                ) from exc
            windows_chain.append(
                (current, pending_windows_owner, pending_windows_owner.identity)
            )
            pending_windows_owner = None
            for part in parts:
                current /= part
                try:
                    pending_windows_owner = _windows_open_bound_handle(
                        current, directory=True
                    )
                except _WindowsHandleAcquisitionError as exc:
                    missing_error = getattr(exc, "winerror", None) in {
                        2,
                        3,
                    } or exc.errno in {2, 3}
                    if not create or not missing_error:
                        raise ValidationError(
                            f"output directory cannot be securely bound: {current}"
                        ) from exc
                    try:
                        current.mkdir()
                    except FileExistsError:
                        pass
                    pending_windows_owner = _windows_open_bound_handle(
                        current, directory=True
                    )
                windows_chain.append(
                    (current, pending_windows_owner, pending_windows_owner.identity)
                )
                pending_windows_owner = None
        else:
            required_dir_fd = (os.open, os.mkdir, os.link, os.unlink)
            if (
                not hasattr(os, "O_DIRECTORY")
                or not hasattr(os, "O_NOFOLLOW")
                or not all(
                    function in os.supports_dir_fd for function in required_dir_fd
                )
            ):
                raise ValidationError(
                    "secure directory-descriptor output binding is unavailable"
                )
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            flags |= getattr(os, "O_CLOEXEC", 0)
            pending_posix_owner = _OwnedDescriptor.open(
                root_absolute,
                flags,
                acquisition_oserror=lambda _exc: ValidationError(
                    f"project root cannot be bound for output: {root_absolute}"
                ),
            )
            root_stat = os.fstat(pending_posix_owner.fileno())
            if not stat_module.S_ISDIR(root_stat.st_mode):
                raise ValidationError("project root is not a directory")
            current = root_absolute
            posix_chain.append(
                (
                    current,
                    pending_posix_owner,
                    (root_stat.st_dev, root_stat.st_ino),
                )
            )
            pending_posix_owner = None
            for part in parts:
                current /= part
                parent_fd = posix_chain[-1][1].fileno()
                try:
                    pending_posix_owner = _OwnedDescriptor.open(
                        part,
                        flags,
                        dir_fd=parent_fd,
                        acquisition_oserror=lambda exc: (
                            exc
                            if isinstance(exc, FileNotFoundError)
                            else ValidationError(
                                "output directory cannot be securely bound: "
                                f"{current}"
                            )
                        ),
                    )
                except FileNotFoundError:
                    if not create:
                        raise ValidationError(
                            f"output directory does not exist: {current}"
                        ) from None
                    try:
                        os.mkdir(part, 0o755, dir_fd=parent_fd)
                    except FileExistsError:
                        pass
                    pending_posix_owner = _OwnedDescriptor.open(
                        part,
                        flags,
                        dir_fd=parent_fd,
                        acquisition_oserror=lambda _exc: ValidationError(
                            f"output directory cannot be securely bound: {current}"
                        ),
                    )
                observed = os.fstat(pending_posix_owner.fileno())
                if not stat_module.S_ISDIR(observed.st_mode):
                    raise ValidationError(f"output path is not a directory: {current}")
                posix_chain.append(
                    (
                        current,
                        pending_posix_owner,
                        (observed.st_dev, observed.st_ino),
                    )
                )
                pending_posix_owner = None
        bound = _BoundOutputDirectory(
            path=directory_absolute,
            posix_chain=posix_chain,
            windows_chain=windows_chain,
        )
        bound.assert_current()
        yield bound
        bound.assert_current()
    except BaseException as exc:
        primary_failure = exc
    if pending_windows_owner is not None:
        _capture_cleanup_failure(
            cleanup_failures,
            "closing unregistered bound output directory handle",
            pending_windows_owner.close,
        )
    for path, handle_owner, _ in reversed(windows_chain):
        _capture_cleanup_failure(
            cleanup_failures,
            f"closing bound output directory handle {path}",
            handle_owner.close,
        )
    if pending_posix_owner is not None:
        _capture_cleanup_failure(
            cleanup_failures,
            "closing unregistered bound output directory descriptor",
            pending_posix_owner.close,
        )
    for path, descriptor_owner, _ in reversed(posix_chain):
        _capture_cleanup_failure(
            cleanup_failures,
            f"closing bound output directory descriptor {path}",
            descriptor_owner.close,
        )
    if primary_failure is not None or cleanup_failures:
        _raise_after_cleanup(primary_failure, cleanup_failures)


def _write_once_artifact(
    path: Path,
    writer: Callable[[Any], Any],
    *,
    root: Path,
) -> tuple[int, int]:
    _, target, parts = _lexical_project_member(path, root)
    if not parts:
        raise ValidationError("artifact path cannot be the project root")
    temporary_name = (
        f".{target.name}.tmp-{uuid.uuid4().hex}" if os.name == "nt" else None
    )
    published_identity: tuple[int, int] | None = None
    temporary_identity: tuple[int, int] | None = None
    returned_successfully = False
    outer_failure: BaseException | None = None
    outer_cleanup_failures: list[CleanupFailure] = []
    try:
        with _bound_output_directory(target.parent, root=root, create=True) as bound:
            primary_failure: BaseException | None = None
            cleanup_failures: list[CleanupFailure] = []
            try:
                with bound.create_file(temporary_name) as handle:
                    writer(handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                    descriptor_stat = os.fstat(handle.fileno())
                    if not stat_module.S_ISREG(descriptor_stat.st_mode):
                        raise ValidationError(
                            "temporary artifact is not a regular file"
                        )
                    temporary_identity = (
                        descriptor_stat.st_dev,
                        descriptor_stat.st_ino,
                    )
                    with _duplicate_descriptor_stream(
                        handle.fileno(), "rb"
                    ) as duplicate:
                        before_digest, _, before_size = _hash_open_handle(duplicate)
                    try:
                        bound.link_open_file(
                            handle,
                            temporary_name,
                            target.name,
                            temporary_identity,
                        )
                    except FileExistsError as exc:
                        raise ValidationError(
                            f"refusing to overwrite write-once artifact: {target}"
                        ) from exc
                    published_identity = temporary_identity
                    with _duplicate_descriptor_stream(
                        handle.fileno(), "rb"
                    ) as duplicate:
                        after_digest, _, after_size = _hash_open_handle(duplicate)
                    if (after_size, after_digest) != (before_size, before_digest):
                        raise ValidationError(
                            "temporary artifact content changed during write-once commit"
                        )
                    if bound.file_identity(target.name) != temporary_identity:
                        raise ValidationError(
                            "write-once destination changed after descriptor-bound linking"
                        )
                    bound.assert_current()
            except BaseException as exc:
                primary_failure = exc
            if primary_failure is not None and published_identity is not None:
                _capture_cleanup_failure(
                    cleanup_failures,
                    "removing incomplete write-once artifact",
                    lambda: bound.unlink_if_identity(target.name, published_identity),
                )
            if temporary_name is not None and temporary_identity is not None:
                _capture_cleanup_failure(
                    cleanup_failures,
                    "removing temporary write-once artifact",
                    lambda: bound.unlink_if_identity(
                        temporary_name, temporary_identity
                    ),
                )
            elif temporary_name is not None:
                _capture_cleanup_failure(
                    cleanup_failures,
                    "removing temporary write-once artifact",
                    lambda: bound.unlink(temporary_name, missing_ok=True),
                )
            if primary_failure is not None or cleanup_failures:
                _raise_after_cleanup(primary_failure, cleanup_failures)
        returned_successfully = True
    except BaseException as exc:
        outer_failure = exc
    if not returned_successfully and published_identity is not None:
        _capture_cleanup_failure(
            outer_cleanup_failures,
            "rolling back published write-once artifact",
            lambda: _secure_unlink_artifact_if_identity(
                target, published_identity, root=root
            ),
        )
    if outer_failure is not None or outer_cleanup_failures:
        _raise_after_cleanup(outer_failure, outer_cleanup_failures)
    if published_identity is None:
        raise ValidationError(
            "write-once artifact returned without a published identity"
        )
    return published_identity


def _secure_unlink_artifact_if_identity(
    path: Path, expected_identity: tuple[int, int], *, root: Path
) -> None:
    _, target, parts = _lexical_project_member(path, root)
    if not parts:
        raise ValidationError("artifact path cannot be the project root")
    with _bound_output_directory(target.parent, root=root, create=False) as bound:
        bound.unlink_if_identity(target.name, expected_identity)
        bound.assert_current()


def _secure_unlink_artifact(path: Path, *, root: Path) -> None:
    _, target, parts = _lexical_project_member(path, root)
    if not parts:
        raise ValidationError("artifact path cannot be the project root")
    with _bound_output_directory(target.parent, root=root, create=False) as bound:
        bound.unlink(target.name, missing_ok=True)
        bound.assert_current()


def _create_bound_directory(path: Path, *, root: Path) -> Path:
    _, target, parts = _lexical_project_member(path, root)
    if not parts:
        raise ValidationError("directory path cannot be the project root")
    with _bound_output_directory(target.parent, root=root, create=True) as parent:
        if os.name == "nt":
            try:
                target.mkdir()
            except FileExistsError as exc:
                raise ValidationError(
                    f"refusing to reuse existing output directory: {target}"
                ) from exc
            with _windows_open_bound_handle(target, directory=True):
                parent.assert_current()
        else:
            try:
                os.mkdir(target.name, 0o755, dir_fd=parent.dir_fd)
            except FileExistsError as exc:
                raise ValidationError(
                    f"refusing to reuse existing output directory: {target}"
                ) from exc
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            with _OwnedDescriptor.open(
                target.name, flags, dir_fd=parent.dir_fd
            ) as child_owner:
                child_stat = os.fstat(child_owner.fileno())
                if not stat_module.S_ISDIR(child_stat.st_mode):
                    raise ValidationError("new output path is not a directory")
                parent.assert_current()
    return target


@contextmanager
def _hold_execution_sources() -> Iterator[dict[str, dict[str, Any]]]:
    """Keep the launcher/controller identities stable across a marker commit."""
    records = _execution_source_records()
    source_paths = {
        _CONTROLLER_SOURCE_RELATIVE: globals().get(_CONTROLLER_PATH_KEY),
        _LAUNCHER_SOURCE_RELATIVE: globals().get(_LAUNCHER_PATH_KEY),
    }
    if not all(isinstance(path, Path) for path in source_paths.values()):
        raise ValidationError("execution source paths are not bound")
    windows_handles: list[tuple[str, Path, _OwnedWindowsHandle, tuple[int, int]]] = []
    posix_descriptors: list[tuple[str, Path, _OwnedDescriptor, tuple[int, int]]] = []
    pending_windows_owner: _OwnedWindowsHandle | None = None
    pending_posix_owner: _OwnedDescriptor | None = None
    primary_failure: BaseException | None = None
    cleanup_failures: list[CleanupFailure] = []
    try:
        for relative, untyped_path in source_paths.items():
            path = cast(Path, untyped_path)
            expected = records[relative]
            if os.name == "nt":
                try:
                    pending_windows_owner = _windows_open_bound_handle(
                        path, directory=False
                    )
                except _WindowsHandleAcquisitionError as exc:
                    raise ValidationError(
                        f"execution source cannot be held stable: {path}"
                    ) from exc
                windows_handles.append(
                    (
                        relative,
                        path,
                        pending_windows_owner,
                        pending_windows_owner.identity,
                    )
                )
                pending_windows_owner = None
            else:
                if not hasattr(os, "O_NOFOLLOW"):
                    raise ValidationError(
                        "no-follow execution-source binding is unavailable"
                    )
                flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
                pending_posix_owner = _OwnedDescriptor.open(
                    path,
                    flags,
                    acquisition_oserror=lambda _exc: ValidationError(
                        f"execution source cannot be held stable: {path}"
                    ),
                )
                observed_stat = os.fstat(pending_posix_owner.fileno())
                if not stat_module.S_ISREG(observed_stat.st_mode):
                    raise ValidationError(
                        f"execution source is not a regular file: {path}"
                    )
                posix_descriptors.append(
                    (
                        relative,
                        path,
                        pending_posix_owner,
                        (observed_stat.st_dev, observed_stat.st_ino),
                    )
                )
                pending_posix_owner = None
                with _duplicate_descriptor_stream(
                    posix_descriptors[-1][2].fileno(), "rb"
                ) as duplicate:
                    digest, _, size = _hash_open_handle(duplicate)
                if size != expected["bytes"] or digest != expected["sha256"]:
                    raise ValidationError(
                        f"{relative} changed while its execution identity was held"
                    )
        _execution_source_records()
        yield records
        _execution_source_records()
        if os.name != "nt":
            for (
                relative,
                path,
                descriptor_owner,
                expected_identity,
            ) in posix_descriptors:
                try:
                    current = os.stat(path, follow_symlinks=False)
                    bound = os.fstat(descriptor_owner.fileno())
                except OSError as exc:
                    raise ValidationError(
                        f"{relative} changed during completion-marker commit"
                    ) from exc
                if (
                    not stat_module.S_ISREG(current.st_mode)
                    or (current.st_dev, current.st_ino) != expected_identity
                    or (bound.st_dev, bound.st_ino) != expected_identity
                ):
                    raise ValidationError(
                        f"{relative} changed during completion-marker commit"
                    )
    except BaseException as exc:
        primary_failure = exc
    if pending_windows_owner is not None:
        _capture_cleanup_failure(
            cleanup_failures,
            "closing unregistered execution source handle",
            pending_windows_owner.close,
        )
    for relative, _, handle_owner, _ in reversed(windows_handles):
        _capture_cleanup_failure(
            cleanup_failures,
            f"closing held execution source {relative}",
            handle_owner.close,
        )
    if pending_posix_owner is not None:
        _capture_cleanup_failure(
            cleanup_failures,
            "closing unregistered execution source descriptor",
            pending_posix_owner.close,
        )
    for relative, _, descriptor_owner, _ in reversed(posix_descriptors):
        _capture_cleanup_failure(
            cleanup_failures,
            f"closing held execution source {relative}",
            descriptor_owner.close,
        )
    if primary_failure is not None or cleanup_failures:
        _raise_after_cleanup(primary_failure, cleanup_failures)


def _referenced_file_records(
    payload: Any,
    *,
    excluded_paths: Collection[str] = (),
) -> list[tuple[str, dict[str, Any]]]:
    """Return every nested file record, rejecting conflicting duplicate paths."""
    discovered: dict[str, tuple[str, dict[str, Any]]] = {}
    excluded = frozenset(excluded_paths)

    def visit(value: Any, label: str) -> None:
        if isinstance(value, Mapping):
            if {"path", "bytes", "sha256"}.issubset(value):
                fields = _record_fields(value, label)
                path = fields["path"]
                byte_count = fields["bytes"]
                digest = fields["sha256"]
                if not isinstance(path, str):
                    raise ValidationError(f"{label} path is not a string")
                if (
                    not isinstance(byte_count, int)
                    or isinstance(byte_count, bool)
                    or byte_count < 0
                ):
                    raise ValidationError(f"{label} byte count is invalid")
                if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                    raise ValidationError(f"{label} has an invalid SHA-256")
                normalized = {
                    "path": path,
                    "bytes": byte_count,
                    "sha256": digest,
                }
                if path not in excluded:
                    prior = discovered.get(path)
                    if prior is not None and prior[1] != normalized:
                        raise ValidationError(
                            f"marker contains conflicting file records for {path}"
                        )
                    discovered[path] = (label, normalized)
            for key, item in value.items():
                visit(item, f"{label}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{label}[{index}]")

    visit(payload, "marker")
    return [discovered[path] for path in sorted(discovered)]


@contextmanager
def _hold_referenced_artifacts(
    payload: Mapping[str, Any], *, root: Path, marker_path: Path
) -> Iterator[None]:
    """Hold and re-hash every artifact named by a marker during its commit."""
    marker_absolute = _project_output_path(marker_path, root=root)
    root_absolute = Path(os.path.abspath(root))
    marker_relative = PurePosixPath(
        marker_absolute.relative_to(root_absolute)
    ).as_posix()
    records = _referenced_file_records(
        payload,
        excluded_paths=(marker_relative,),
    )
    windows_handles: list[
        tuple[
            str,
            Path,
            _OwnedWindowsHandle,
            tuple[int, int],
            dict[str, Any],
        ]
    ] = []
    posix_descriptors: list[
        tuple[str, Path, _OwnedDescriptor, tuple[int, int], dict[str, Any]]
    ] = []
    pending_windows_owner: _OwnedWindowsHandle | None = None
    pending_posix_owner: _OwnedDescriptor | None = None
    primary_failure: BaseException | None = None
    cleanup_failures: list[CleanupFailure] = []
    try:
        for label, expected in records:
            path = _verify_record(expected, root=root, label=label)
            if os.name == "nt":
                try:
                    pending_windows_owner = _windows_open_bound_handle(
                        path, directory=False
                    )
                except _WindowsHandleAcquisitionError as exc:
                    raise ValidationError(
                        f"referenced artifact cannot be held stable: {path}"
                    ) from exc
                windows_handles.append(
                    (
                        label,
                        path,
                        pending_windows_owner,
                        pending_windows_owner.identity,
                        expected,
                    )
                )
                pending_windows_owner = None
                # The no-write/no-delete sharing mode is now held.  Re-verify so
                # the bytes belong to the held path instance.
                _verify_record(expected, root=root, label=label)
            else:
                if not hasattr(os, "O_NOFOLLOW"):
                    raise ValidationError(
                        "no-follow referenced-artifact binding is unavailable"
                    )
                flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
                pending_posix_owner = _OwnedDescriptor.open(
                    path,
                    flags,
                    acquisition_oserror=lambda _exc: ValidationError(
                        f"referenced artifact cannot be held stable: {path}"
                    ),
                )
                observed = os.fstat(pending_posix_owner.fileno())
                if not stat_module.S_ISREG(observed.st_mode):
                    raise ValidationError(
                        f"referenced artifact is not a regular file: {path}"
                    )
                identity = (observed.st_dev, observed.st_ino)
                current = os.stat(path, follow_symlinks=False)
                if (
                    not stat_module.S_ISREG(current.st_mode)
                    or (current.st_dev, current.st_ino) != identity
                ):
                    raise ValidationError(
                        f"referenced artifact changed while being held: {path}"
                    )
                with _duplicate_descriptor_stream(
                    pending_posix_owner.fileno(), "rb"
                ) as duplicate:
                    digest, _, size = _hash_open_handle(duplicate)
                if size != expected["bytes"] or digest != expected["sha256"]:
                    raise ValidationError(
                        f"referenced artifact changed while being held: {path}"
                    )
                posix_descriptors.append(
                    (label, path, pending_posix_owner, identity, expected)
                )
                pending_posix_owner = None
        yield
        if os.name == "nt":
            for label, path, _, expected_identity, expected in windows_handles:
                _verify_record(expected, root=root, label=label)
                with _windows_open_bound_handle(path, directory=False) as current_owner:
                    if current_owner.identity != expected_identity:
                        raise ValidationError(
                            f"referenced artifact changed during marker commit: {path}"
                        )
        else:
            for (
                label,
                path,
                descriptor_owner,
                expected_identity,
                expected,
            ) in posix_descriptors:
                try:
                    current = os.stat(path, follow_symlinks=False)
                    bound = os.fstat(descriptor_owner.fileno())
                except OSError as exc:
                    raise ValidationError(
                        f"referenced artifact changed during marker commit: {path}"
                    ) from exc
                if (
                    not stat_module.S_ISREG(current.st_mode)
                    or (current.st_dev, current.st_ino) != expected_identity
                    or (bound.st_dev, bound.st_ino) != expected_identity
                ):
                    raise ValidationError(
                        f"referenced artifact changed during marker commit: {path}"
                    )
                with _duplicate_descriptor_stream(
                    descriptor_owner.fileno(), "rb"
                ) as duplicate:
                    digest, _, size = _hash_open_handle(duplicate)
                if size != expected["bytes"] or digest != expected["sha256"]:
                    raise ValidationError(
                        f"referenced artifact changed during marker commit: {path}"
                    )
    except BaseException as exc:
        primary_failure = exc
    if pending_windows_owner is not None:
        _capture_cleanup_failure(
            cleanup_failures,
            "closing unregistered referenced artifact handle",
            pending_windows_owner.close,
        )
    for label, _, handle_owner, _, _ in reversed(windows_handles):
        _capture_cleanup_failure(
            cleanup_failures,
            f"closing held referenced artifact {label}",
            handle_owner.close,
        )
    if pending_posix_owner is not None:
        _capture_cleanup_failure(
            cleanup_failures,
            "closing unregistered referenced artifact descriptor",
            pending_posix_owner.close,
        )
    for label, _, descriptor_owner, _, _ in reversed(posix_descriptors):
        _capture_cleanup_failure(
            cleanup_failures,
            f"closing held referenced artifact {label}",
            descriptor_owner.close,
        )
    if primary_failure is not None or cleanup_failures:
        _raise_after_cleanup(primary_failure, cleanup_failures)


def _commit_execution_bound_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    root: Path,
) -> None:
    """Commit a marker while its sources and referenced artifacts remain bound."""
    published_identity: tuple[int, int] | None = None
    primary_failure: BaseException | None = None
    cleanup_failures: list[CleanupFailure] = []
    try:
        with _hold_referenced_artifacts(payload, root=root, marker_path=path):
            with _hold_execution_sources():
                published_identity = _atomic_json(path, payload, root=root)
    except BaseException as exc:
        primary_failure = exc
    if primary_failure is not None and published_identity is not None:
        _capture_cleanup_failure(
            cleanup_failures,
            "removing marker committed under a failed source binding",
            lambda: _secure_unlink_artifact_if_identity(
                path, published_identity, root=root
            ),
        )
    if primary_failure is not None or cleanup_failures:
        _raise_after_cleanup(primary_failure, cleanup_failures)


def _record_file(path: Path, root: Path = ROOT) -> dict[str, Any]:
    digest, size = _stable_hash_and_size(path, "sha256")
    return {
        "path": _portable_relative(path, root),
        "bytes": size,
        "sha256": digest,
    }


def _controller_execution_source_record() -> dict[str, Any]:
    """Verify and return the exact source record used for controller execution."""
    bound = globals().get(_CONTROLLER_RECORD_KEY)
    bound_path = globals().get(_CONTROLLER_PATH_KEY)
    if not isinstance(bound, Mapping) or not isinstance(bound_path, Path):
        raise ValidationError("controller execution source is not hash-bound")
    record = _record_fields(bound, "controller execution source")
    if (
        record["path"] != _CONTROLLER_SOURCE_RELATIVE
        or not isinstance(record["bytes"], int)
        or isinstance(record["bytes"], bool)
        or record["bytes"] < 0
        or not isinstance(record["sha256"], str)
        or not SHA256_RE.fullmatch(record["sha256"])
    ):
        raise ValidationError("controller execution source binding is invalid")
    _, observed = _stable_read_bytes(
        bound_path,
        label="controller execution source",
    )
    observed["path"] = _CONTROLLER_SOURCE_RELATIVE
    _assert_same_record(observed, record, label="controller execution source")
    return dict(record)


def _launcher_execution_source_record() -> dict[str, Any]:
    """Verify and return the exact launcher source bound before controller exec."""
    bound = globals().get(_LAUNCHER_RECORD_KEY)
    bound_path = globals().get(_LAUNCHER_PATH_KEY)
    if not isinstance(bound, Mapping) or not isinstance(bound_path, Path):
        raise ValidationError("launcher execution source is not hash-bound")
    record = _record_fields(bound, "launcher execution source")
    if (
        record["path"] != _LAUNCHER_SOURCE_RELATIVE
        or not isinstance(record["bytes"], int)
        or isinstance(record["bytes"], bool)
        or record["bytes"] < 0
        or not isinstance(record["sha256"], str)
        or not SHA256_RE.fullmatch(record["sha256"])
    ):
        raise ValidationError("launcher execution source binding is invalid")
    _, observed = _stable_read_bytes(
        bound_path,
        label="launcher execution source",
    )
    observed["path"] = _LAUNCHER_SOURCE_RELATIVE
    _assert_same_record(observed, record, label="launcher execution source")
    return dict(record)


def _execution_source_records() -> dict[str, dict[str, Any]]:
    """Verify both members of the trusted execution bootstrap."""
    return {
        _CONTROLLER_SOURCE_RELATIVE: _controller_execution_source_record(),
        _LAUNCHER_SOURCE_RELATIVE: _launcher_execution_source_record(),
    }


def _safe_project_path(relative: str, root: Path = ROOT) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValidationError(f"unsafe project-relative path: {relative!r}")
    candidate = PurePosixPath(relative)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or not candidate.parts
        or relative != candidate.as_posix()
        or ":" in candidate.parts[0]
    ):
        raise ValidationError(f"unsafe project-relative path: {relative!r}")
    resolved = (root / Path(*candidate.parts)).resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValidationError(f"path escapes project root: {relative!r}") from exc
    return resolved


def _verify_record(record: Mapping[str, Any], *, root: Path = ROOT, label: str) -> Path:
    required = {"path", "bytes", "sha256"}
    if not isinstance(record, Mapping) or not required.issubset(record):
        raise ValidationError(f"{label} file record is incomplete")
    if not isinstance(record["path"], str):
        raise ValidationError(f"{label} path is not a string")
    if (
        not isinstance(record["bytes"], int)
        or isinstance(record["bytes"], bool)
        or record["bytes"] < 0
    ):
        raise ValidationError(f"{label} byte count is invalid")
    if not isinstance(record["sha256"], str):
        raise ValidationError(f"{label} SHA-256 is not a string")
    path = _safe_project_path(record["path"], root)
    if not path.is_file():
        raise ValidationError(f"{label} is not a regular file")
    digest = record["sha256"]
    if not SHA256_RE.fullmatch(digest):
        raise ValidationError(f"{label} has an invalid SHA-256")
    observed_digest, observed_size = _stable_hash_and_size(path, "sha256")
    if observed_size != record["bytes"] or observed_digest != digest:
        raise ValidationError(f"{label} changed after binding: {path}")
    return path


def _atomic_bytes(path: Path, payload: bytes, *, root: Path) -> tuple[int, int]:
    return _write_once_artifact(path, lambda handle: handle.write(payload), root=root)


def _atomic_json(
    path: Path, payload: Mapping[str, Any], *, root: Path
) -> tuple[int, int]:
    return _atomic_bytes(path, _canonical_json_bytes(payload), root=root)


def _atomic_tsv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    root: Path,
) -> tuple[int, int]:
    return _atomic_bytes(path, _tsv_bytes(rows, fields), root=root)


def _tsv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(fields), delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _read_lines(path: Path) -> list[str]:
    payload, _ = _stable_read_bytes(path, label="identifier manifest")
    return _read_lines_bytes(payload, path)


def _read_lines_bytes(payload: bytes, path: Path) -> list[str]:
    try:
        raw = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValidationError(f"manifest is not valid UTF-8: {path}") from exc
    values = [line.strip() for line in raw]
    if raw != values:
        raise ValidationError(f"manifest contains surrounding whitespace: {path}")
    if (
        not values
        or any(not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValidationError(f"manifest must contain unique non-empty values: {path}")
    return values


def _csv_header(path: Path, label: str) -> list[str]:
    payload, _ = _stable_read_bytes(path, label=f"{label} CSV")
    return _csv_header_bytes(payload, label)


def _csv_header_bytes(payload: bytes, label: str) -> list[str]:
    try:
        text = payload.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text, newline="")))
    except (StopIteration, UnicodeDecodeError, csv.Error) as exc:
        raise ValidationError(f"{label} has no readable CSV header") from exc
    return _require_unique(header, f"{label} column")


def _clean_identifiers(values: Iterable[Any], label: str) -> list[str]:
    output: list[str] = []
    for value in values:
        try:
            missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = False
        raw = "" if missing else str(value)
        cleaned = raw.strip()
        if raw != cleaned:
            raise ValidationError(f"{label} contains surrounding whitespace")
        if not cleaned:
            raise ValidationError(f"{label} contains an empty identifier")
        output.append(cleaned)
    return output


def _require_unique(values: Iterable[Any], label: str) -> list[str]:
    output = _clean_identifiers(values, label)
    duplicates = [key for key, count in Counter(output).items() if count > 1]
    if duplicates:
        raise ValidationError(
            f"{label} contains duplicate identifiers: {duplicates[:3]}"
        )
    return output


def _ordered_axis_binding(
    values: Iterable[Any], *, label: str, source: str
) -> dict[str, Any]:
    identifiers = _require_unique(values, label)
    return {
        "source": source,
        "count": len(identifiers),
        "ordered_identifiers_sha256": hashlib.sha256(
            _canonical_json_bytes(identifiers)
        ).hexdigest(),
    }


def _prediction_axis_bindings(
    gene_ids: Sequence[str], sample_ids: Sequence[str]
) -> dict[str, Any]:
    return {
        "genes": _ordered_axis_binding(
            gene_ids,
            label="prediction gene",
            source="frozen data/processed/genes.txt line order",
        ),
        "eligible_organoid_samples": _ordered_axis_binding(
            sample_ids,
            label="eligible organoid sample_ID",
            source="all and only frozen RNA sample_ID rows in publisher file order after required covariate and driver-mutation membership checks",
        ),
    }


def _finite_numeric(frame: pd.DataFrame, label: str) -> np.ndarray:
    try:
        values = frame.to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} contains non-numeric values") from exc
    if not np.isfinite(values).all():
        raise ValidationError(f"{label} contains NaN or infinite values")
    return values


def _load_figshare_metadata(path: Path) -> dict[str, Any]:
    payload = _load_json_mapping(path, "Figshare metadata")
    return _validate_figshare_metadata(payload)


def _load_figshare_metadata_bytes(payload_bytes: bytes) -> dict[str, Any]:
    return _validate_figshare_metadata(
        _parse_json_mapping(payload_bytes, "Figshare metadata")
    )


def _validate_figshare_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("id") != FIGSHARE_ITEM_ID or payload.get("doi") != FIGSHARE_DOI:
        raise ValidationError("unexpected Figshare item identity")
    if payload.get("version") != 1 or payload.get("is_public") is not True:
        raise ValidationError("Figshare item must be public version 1")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValidationError("Figshare file inventory is missing")
    return payload


def _figshare_file(payload: Mapping[str, Any], file_id: int) -> dict[str, Any]:
    matches = [
        item
        for item in payload["files"]
        if isinstance(item, dict) and item.get("id") == file_id
    ]
    if len(matches) != 1:
        raise ValidationError(
            f"Figshare metadata does not contain exactly one file id {file_id}"
        )
    item = matches[0]
    supplied = str(item.get("supplied_md5", "")).lower()
    computed = str(item.get("computed_md5", "")).lower()
    if not MD5_RE.fullmatch(supplied) or supplied != computed:
        raise ValidationError(f"Figshare MD5 metadata is invalid for file id {file_id}")
    return {
        "id": int(item["id"]),
        "name": str(item["name"]),
        "bytes": int(item["size"]),
        "md5": supplied,
        "download_url": str(item["download_url"]),
        "mimetype": str(item.get("mimetype", "")),
    }


def _normalise_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def _trusted_site_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for key in ("purelib", "platlib"):
        candidate = sysconfig.get_paths().get(key)
        if not candidate:
            continue
        try:
            resolved = Path(candidate).resolve(strict=True)
        except OSError as exc:
            raise ValidationError(
                f"trusted Python installation path is unavailable: {key}"
            ) from exc
        if resolved not in roots:
            roots.append(resolved)
    if not roots:
        raise ValidationError(
            "trusted Python package installation paths are unavailable"
        )
    return tuple(roots)


def _trusted_runtime_import_roots() -> tuple[Path, ...]:
    """Return the only filesystem roots usable during scientific imports."""
    roots = list(_trusted_site_roots())
    for key in ("stdlib", "platstdlib"):
        candidate = sysconfig.get_paths().get(key)
        if not candidate:
            continue
        try:
            resolved = Path(candidate).resolve(strict=True)
        except OSError as exc:
            raise ValidationError(
                f"trusted Python installation path is unavailable: {key}"
            ) from exc
        if resolved not in roots:
            roots.append(resolved)
        dynamic_directory = resolved / "lib-dynload"
        if dynamic_directory.exists() or dynamic_directory.is_symlink():
            try:
                resolved_dynamic = dynamic_directory.resolve(strict=True)
            except OSError as exc:
                raise ValidationError(
                    f"trusted Python dynamic library path is unavailable: {key}"
                ) from exc
            if not resolved_dynamic.is_dir() or not resolved_dynamic.is_relative_to(
                resolved
            ):
                raise ValidationError(
                    f"trusted Python dynamic library path escaped stdlib: {key}"
                )
            if resolved_dynamic not in roots:
                roots.append(resolved_dynamic)
    dll_directory = Path(sys.base_prefix) / "DLLs"
    if dll_directory.is_dir():
        resolved_dlls = dll_directory.resolve(strict=True)
        if resolved_dlls not in roots:
            roots.append(resolved_dlls)
    return tuple(roots)


def _path_belongs_to_roots(path: Path, roots: Sequence[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _assert_transitive_module_origin(
    module: Any, *, name: str, roots: Sequence[Path]
) -> None:
    """Reject imported dependency modules outside trusted Python roots."""
    if not isinstance(module, types.ModuleType):
        return
    module_spec = getattr(module, "__spec__", None)
    origin = getattr(module_spec, "origin", None)
    if origin in {"built-in", "frozen"}:
        return
    if isinstance(origin, str):
        try:
            resolved = Path(origin).resolve(strict=True)
        except OSError as exc:
            raise ValidationError(
                f"transitive runtime module origin cannot be verified: {name}"
            ) from exc
        if not _path_belongs_to_roots(resolved, roots):
            raise ValidationError(
                f"transitive runtime module escaped trusted import roots: {name}"
            )
        return
    locations = getattr(module_spec, "submodule_search_locations", None)
    if locations:
        for location in locations:
            try:
                resolved_location = Path(location).resolve(strict=True)
            except OSError as exc:
                raise ValidationError(
                    f"transitive runtime namespace cannot be verified: {name}"
                ) from exc
            if not _path_belongs_to_roots(resolved_location, roots):
                raise ValidationError(
                    f"transitive runtime namespace escaped trusted import roots: {name}"
                )
        return
    top_level = name.partition(".")[0]
    top_module = sys.modules.get(top_level)
    if top_module is module:
        raise ValidationError(f"transitive runtime module has no origin: {name}")
    _assert_transitive_module_origin(top_module, name=top_level, roots=roots)


def _is_new_cython_runtime_module(
    module: Any, *, name: str, modules_snapshot: Mapping[str, Any]
) -> bool:
    """Recognise Cython's originless runtime modules created during this import."""
    if name == "cython_runtime":
        expected_keys = {
            "__name__",
            "__doc__",
            "__package__",
            "__loader__",
            "__spec__",
        }
    elif re.fullmatch(r"_cython_[0-9]+_[0-9]+_[0-9]+", name):
        expected_keys = {
            "__name__",
            "__doc__",
            "__package__",
            "__loader__",
            "__spec__",
            "_common_types_metatype",
            "generator",
            "cython_function_or_method",
            "fused_cython_function",
        }
    else:
        return False
    return (
        name not in modules_snapshot
        and isinstance(module, types.ModuleType)
        and getattr(module, "__spec__", None) is None
        and getattr(module, "__file__", None) is None
        and set(vars(module)).issubset(expected_keys)
    )


def _install_runtime_import_path(path_object: list[str], roots: Sequence[Path]) -> None:
    """Install the isolated import path as one recoverable transaction step."""
    path_object[:] = [str(root) for root in roots]
    sys.path = path_object


def _install_runtime_import_hook(import_hook: Callable[..., Any]) -> None:
    builtins.__import__ = import_hook


def _restore_runtime_import_hook(original_import: Callable[..., Any]) -> None:
    builtins.__import__ = original_import


def _restore_sys_path(
    original_object: list[str], original_values: Sequence[str]
) -> None:
    original_object[:] = original_values
    sys.path = original_object


def _restore_mapping_state(
    target: MutableMapping[str, Any], snapshot: Mapping[str, Any]
) -> None:
    """Restore exact keys and value identities without replacing ``target``."""
    for name in tuple(target):
        if name not in snapshot:
            target.pop(name, None)
    for name, value in snapshot.items():
        if name not in target or target[name] is not value:
            target[name] = value


def _restore_runtime_sys_modules(
    original_object: dict[str, Any], snapshot: Mapping[str, Any]
) -> None:
    _restore_mapping_state(original_object, snapshot)
    sys.modules = original_object


@contextmanager
def _trusted_runtime_import_scope() -> Iterator[None]:
    """Isolate third-party execution from PYTHONPATH and audit imported modules."""
    roots = _trusted_runtime_import_roots()
    original_path_object = sys.path
    path_snapshot = list(original_path_object)
    original_modules_object = sys.modules
    modules_snapshot = dict(original_modules_object)
    original_import = builtins.__import__
    primary_failure: BaseException | None = None
    cleanup_failures: list[CleanupFailure] = []
    restoration_steps: tuple[tuple[str, Callable[[], None]], ...] = (
        (
            "runtime import hook restoration",
            lambda: _restore_runtime_import_hook(original_import),
        ),
        (
            "runtime import path restoration",
            lambda: _restore_sys_path(original_path_object, path_snapshot),
        ),
    )

    def trusted_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        imported = original_import(name, globals, locals, fromlist, level)
        imported_name = getattr(imported, "__name__", name)
        _assert_transitive_module_origin(imported, name=imported_name, roots=roots)
        if isinstance(imported_name, str):
            for member in fromlist or ():
                if member == "*":
                    continue
                candidate_name = f"{imported_name}.{member}"
                candidate = sys.modules.get(candidate_name)
                if candidate is not None:
                    _assert_transitive_module_origin(
                        candidate, name=candidate_name, roots=roots
                    )
        return imported

    try:
        try:
            _install_runtime_import_path(original_path_object, roots)
            _install_runtime_import_hook(trusted_import)
            yield
            for name, module in tuple(sys.modules.items()):
                if modules_snapshot.get(name) is module:
                    continue
                if _is_new_cython_runtime_module(
                    module, name=name, modules_snapshot=modules_snapshot
                ):
                    continue
                _assert_transitive_module_origin(module, name=name, roots=roots)
        except BaseException as exc:
            primary_failure = exc
    finally:
        try:
            for label, cleanup in restoration_steps:
                _capture_cleanup_failure(
                    cleanup_failures,
                    f"initial {label}",
                    cleanup,
                )
        except BaseException as exc:
            if primary_failure is None:
                primary_failure = exc
            else:
                cleanup_failures.append(("initial runtime restoration transition", exc))
        try:
            for label, cleanup in restoration_steps:
                _capture_cleanup_failure(
                    cleanup_failures,
                    f"second-chance {label}",
                    cleanup,
                )
        except BaseException as exc:
            if primary_failure is None:
                primary_failure = exc
            else:
                cleanup_failures.append(
                    ("second-chance runtime restoration transition", exc)
                )
        if primary_failure is not None or cleanup_failures:
            try:
                _capture_cleanup_failure(
                    cleanup_failures,
                    "initial runtime sys.modules restoration",
                    lambda: _restore_runtime_sys_modules(
                        original_modules_object, modules_snapshot
                    ),
                )
            except BaseException as exc:
                cleanup_failures.append(
                    ("initial runtime sys.modules restoration transition", exc)
                )
            try:
                _capture_cleanup_failure(
                    cleanup_failures,
                    "second-chance runtime sys.modules restoration",
                    lambda: _restore_runtime_sys_modules(
                        original_modules_object, modules_snapshot
                    ),
                )
            except BaseException as exc:
                cleanup_failures.append(
                    (
                        "second-chance runtime sys.modules restoration transition",
                        exc,
                    )
                )
            _raise_after_cleanup(primary_failure, cleanup_failures)


def _trusted_distribution(name: str) -> importlib.metadata.Distribution:
    canonical = _normalise_distribution_name(name)
    matches: list[importlib.metadata.Distribution] = []
    seen_locations: set[Path] = set()
    for root in _trusted_site_roots():
        for distribution in importlib.metadata.distributions(path=[str(root)]):
            observed_name = str(distribution.metadata.get("Name") or distribution.name)
            if _normalise_distribution_name(observed_name) != canonical:
                continue
            try:
                location = Path(distribution.locate_file("")).resolve(strict=True)
            except OSError as exc:
                raise ValidationError(
                    f"runtime distribution location cannot be resolved: {name}"
                ) from exc
            if location not in seen_locations:
                seen_locations.add(location)
                matches.append(distribution)
    if len(matches) != 1:
        raise ValidationError(
            f"required runtime distribution is not installed exactly once in the "
            f"trusted environment: {name}"
        )
    return matches[0]


def _trusted_runtime_spec(
    module_name: str,
) -> tuple[importlib.machinery.ModuleSpec, Path]:
    distribution_name = RUNTIME_MODULE_DISTRIBUTIONS.get(module_name)
    if distribution_name is None:
        raise ValidationError(f"runtime module is not allowlisted: {module_name}")
    distribution = _trusted_distribution(distribution_name)
    try:
        distribution_root = Path(distribution.locate_file("")).resolve(strict=True)
    except OSError as exc:
        raise ValidationError(
            f"runtime distribution root cannot be resolved: {distribution_name}"
        ) from exc
    search_locations: Sequence[str] = (str(distribution_root),)
    current_name = ""
    spec: importlib.machinery.ModuleSpec | None = None
    for component in module_name.split("."):
        current_name = component if not current_name else f"{current_name}.{component}"
        spec = importlib.machinery.PathFinder.find_spec(current_name, search_locations)
        if spec is None:
            raise ValidationError(
                f"runtime module is absent from its installed distribution: {module_name}"
            )
        if current_name != module_name:
            locations = spec.submodule_search_locations
            if not locations:
                raise ValidationError(
                    f"runtime module parent is not a package: {current_name}"
                )
            search_locations = tuple(str(value) for value in locations)
    assert spec is not None
    if spec.loader is None or not isinstance(spec.origin, str):
        raise ValidationError(
            f"runtime module lacks an executable origin: {module_name}"
        )
    try:
        origin = Path(spec.origin).resolve(strict=True)
        relative_origin = origin.relative_to(distribution_root).as_posix()
    except (OSError, ValueError) as exc:
        raise ValidationError(
            f"runtime module origin cannot be bound to its distribution: {module_name}"
        ) from exc
    distribution_files = {
        str(item).replace("\\", "/") for item in (distribution.files or ())
    }
    if relative_origin not in distribution_files:
        raise ValidationError(
            f"runtime module origin is absent from its distribution inventory: {module_name}"
        )
    for location in spec.submodule_search_locations or ():
        try:
            Path(location).resolve(strict=True).relative_to(distribution_root)
        except (OSError, ValueError) as exc:
            raise ValidationError(
                f"runtime package search path escapes its distribution: {module_name}"
            ) from exc
    return spec, origin


def _assert_runtime_module_origin(module: Any, *, name: str, origin: Path) -> None:
    module_spec = getattr(module, "__spec__", None)
    reported_origin = getattr(module_spec, "origin", None)
    reported_file = getattr(module, "__file__", None)
    try:
        resolved_spec = Path(reported_origin).resolve(strict=True)
        resolved_file = Path(reported_file).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ValidationError(
            f"preloaded runtime module origin cannot be verified: {name}"
        ) from exc
    if resolved_spec != origin or resolved_file != origin:
        raise ValidationError(
            f"refusing path-shadowed or preloaded runtime module: {name}"
        )


_RUNTIME_FINGERPRINT_MAX_DEPTH = 24
_RUNTIME_FINGERPRINT_MAX_NODES = 2_048
_RUNTIME_FINGERPRINT_MAX_ITEMS = 512
_RUNTIME_FINGERPRINT_MAX_BYTES = 2 * 1024 * 1024
_RUNTIME_MISSING = object()


@dataclass
class _RuntimeFingerprintContext:
    expected_bindings: Mapping[str, Any] | None
    seen: dict[int, int]
    bindings: dict[str, Any]
    node_count: int = 0
    byte_count: int = 0

    def touch(self, path: str, depth: int) -> None:
        if depth > _RUNTIME_FINGERPRINT_MAX_DEPTH:
            raise ValidationError(
                "trusted runtime fingerprint exceeded its depth limit"
            )
        self.node_count += 1
        if self.node_count > _RUNTIME_FINGERPRINT_MAX_NODES:
            raise ValidationError("trusted runtime fingerprint exceeded its node limit")
        self.consume_text(path)

    def consume_text(self, value: str) -> None:
        if len(value) > _RUNTIME_FINGERPRINT_MAX_BYTES:
            raise ValidationError("trusted runtime fingerprint text is too large")
        self.byte_count += len(value.encode("utf-8"))
        if self.byte_count > _RUNTIME_FINGERPRINT_MAX_BYTES:
            raise ValidationError("trusted runtime fingerprint exceeded its byte limit")

    def bind(self, path: str, value: Any, depth: int) -> tuple[int, bool]:
        self.touch(path, depth)
        if self.expected_bindings is not None:
            expected = self.expected_bindings.get(path, _RUNTIME_MISSING)
            if expected is _RUNTIME_MISSING or expected is not value:
                raise ValidationError(
                    f"trusted runtime object binding changed at {path}"
                )
        self.bindings[path] = value
        object_id = id(value)
        existing = self.seen.get(object_id)
        if existing is not None:
            return existing, True
        node = len(self.seen)
        self.seen[object_id] = node
        return node, False


def _runtime_code_names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for constant in code.co_consts:
        if type(constant) is types.CodeType:
            names.update(_runtime_code_names(constant))
    return names


def _runtime_child_path(path: str, category: str, name: str) -> str:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return f"{path}.{category}[{digest}]"


def _runtime_direct_attribute(value: Any, name: str) -> Any:
    try:
        if isinstance(value, type):
            return type.__getattribute__(value, name)
        return object.__getattribute__(value, name)
    except (AttributeError, TypeError):
        return _RUNTIME_MISSING


def _runtime_metadata_text(value: Any, name: str, *, path: str) -> str | None:
    observed = _runtime_direct_attribute(value, name)
    if observed is _RUNTIME_MISSING or observed is None:
        return None
    if type(observed) is not str:
        raise ValidationError(f"trusted runtime metadata is invalid at {path}.{name}")
    return observed


def _runtime_guard_code(
    code: types.CodeType,
    *,
    context: _RuntimeFingerprintContext,
    path: str,
    depth: int,
) -> None:
    context.touch(path, depth)
    collections = (
        code.co_consts,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
    )
    if any(len(values) > _RUNTIME_FINGERPRINT_MAX_ITEMS for values in collections):
        raise ValidationError("trusted runtime code exceeded its collection limit")
    context.byte_count += len(code.co_code)
    context.consume_text(code.co_filename)
    context.consume_text(code.co_name)
    context.consume_text(code.co_qualname)
    if context.byte_count > _RUNTIME_FINGERPRINT_MAX_BYTES:
        raise ValidationError("trusted runtime code exceeded its byte limit")
    for index, constant in enumerate(code.co_consts):
        if type(constant) is types.CodeType:
            _runtime_guard_code(
                constant,
                context=context,
                path=f"{path}.constant[{index}]",
                depth=depth + 1,
            )


def _runtime_reference_leaf(
    value: Any,
    *,
    kind: str,
    context: _RuntimeFingerprintContext,
    path: str,
    depth: int,
) -> dict[str, Any]:
    node, repeated = context.bind(path, value, depth)
    if repeated:
        return {"kind": "reference", "node": node}
    return {"kind": kind, "node": node}


def _runtime_module_descriptor(
    value: types.ModuleType,
    *,
    context: _RuntimeFingerprintContext,
    path: str,
    depth: int,
    node: int | None = None,
) -> dict[str, Any]:
    if node is None:
        node, repeated = context.bind(path, value, depth)
        if repeated:
            return {"kind": "reference", "node": node}
    descriptor: dict[str, Any] = {"kind": "module", "node": node}
    namespace = object.__getattribute__(value, "__dict__")
    if type(namespace) is not dict:
        raise ValidationError(f"trusted runtime module namespace is invalid at {path}")
    name = namespace.get("__name__")
    filename = namespace.get("__file__")
    if type(name) is not str or filename is not None and type(filename) is not str:
        raise ValidationError(f"trusted runtime module metadata is invalid at {path}")
    context.consume_text(name)
    if filename is not None:
        context.consume_text(filename)
    spec = namespace.get("__spec__")
    origin: str | None = None
    if spec is not None:
        spec_binding = _runtime_reference_leaf(
            spec,
            kind="module_spec",
            context=context,
            path=f"{path}.__spec__",
            depth=depth + 1,
        )
        raw_origin = _runtime_direct_attribute(spec, "origin")
        if raw_origin is not _RUNTIME_MISSING and raw_origin is not None:
            if type(raw_origin) is not str:
                raise ValidationError(
                    f"trusted runtime module origin is invalid at {path}"
                )
            origin = raw_origin
            context.consume_text(origin)
        descriptor["spec"] = spec_binding
    descriptor.update({"name": name, "file": filename, "origin": origin})
    return descriptor


def _runtime_executable_descriptor(
    value: Any,
    *,
    code: types.CodeType,
    namespace: dict[str, Any] | None,
    context: _RuntimeFingerprintContext,
    path: str,
    depth: int,
    node: int,
) -> dict[str, Any]:
    module_name = _runtime_metadata_text(value, "__module__", path=path)
    qualname = _runtime_metadata_text(value, "__qualname__", path=path)
    function_name = _runtime_metadata_text(value, "__name__", path=path)
    for metadata in (module_name, qualname, function_name):
        if metadata is not None:
            context.consume_text(metadata)
    _runtime_guard_code(code, context=context, path=f"{path}.__code__", depth=depth + 1)

    descriptor: dict[str, Any] = {
        "kind": "executable",
        "node": node,
        "module": module_name,
        "qualname": qualname,
        "name": function_name,
        "type_module": type(value).__module__,
        "type_qualname": type(value).__qualname__,
        "code": _trusted_code_descriptor(code),
    }
    for attribute in ("__defaults__", "__kwdefaults__", "__annotations__", "__dict__"):
        state = _runtime_direct_attribute(value, attribute)
        if state is _RUNTIME_MISSING:
            state = None
        descriptor[attribute] = _runtime_execution_descriptor(
            state,
            context=context,
            path=f"{path}.{attribute}",
            depth=depth + 1,
        )

    closure = _runtime_direct_attribute(value, "__closure__")
    if closure is _RUNTIME_MISSING or closure is None:
        descriptor["closure"] = None
    elif type(closure) is tuple:
        if len(closure) > _RUNTIME_FINGERPRINT_MAX_ITEMS:
            raise ValidationError("trusted runtime closure exceeded its item limit")
        cells: list[Any] = []
        for index, cell in enumerate(closure):
            try:
                cell_value = cell.cell_contents
            except ValueError:
                cells.append({"kind": "empty_cell"})
            else:
                cells.append(
                    _runtime_execution_descriptor(
                        cell_value,
                        context=context,
                        path=f"{path}.__closure__[{index}]",
                        depth=depth + 1,
                    )
                )
        descriptor["closure"] = cells
    else:
        raise ValidationError(f"trusted runtime closure is invalid at {path}")

    if not (
        isinstance(module_name, str)
        and (module_name == "pyreadr" or module_name.startswith("pyreadr."))
    ):
        descriptor["globals"] = {"kind": "external_namespace"}
        return descriptor
    if type(namespace) is not dict:
        raise ValidationError(f"trusted runtime globals are invalid at {path}")

    globals_binding = _runtime_reference_leaf(
        namespace,
        kind="globals",
        context=context,
        path=f"{path}.__globals__",
        depth=depth + 1,
    )
    raw_builtins = namespace.get("__builtins__", builtins)
    if type(raw_builtins) is dict:
        builtin_namespace = raw_builtins
    elif type(raw_builtins) is types.ModuleType and raw_builtins is builtins:
        builtin_namespace = vars(builtins)
    else:
        raise ValidationError(f"trusted runtime builtins are invalid at {path}")
    builtins_binding = _runtime_reference_leaf(
        raw_builtins,
        kind="builtins",
        context=context,
        path=f"{path}.__builtins__",
        depth=depth + 1,
    )

    names = sorted(_runtime_code_names(code))
    if len(names) > _RUNTIME_FINGERPRINT_MAX_ITEMS:
        raise ValidationError("trusted runtime globals exceeded their item limit")
    dependencies: dict[str, Any] = {}
    for name in names:
        context.consume_text(name)
        if name in namespace:
            source = "global"
            dependency = namespace[name]
        elif name in builtin_namespace:
            source = "builtin"
            dependency = builtin_namespace[name]
        else:
            dependencies[name] = {"source": "missing"}
            continue
        dependencies[name] = {
            "source": source,
            "value": _runtime_execution_descriptor(
                dependency,
                context=context,
                path=_runtime_child_path(path, "global", name),
                depth=depth + 1,
            ),
        }
    descriptor["globals"] = {
        "namespace": globals_binding,
        "builtins": builtins_binding,
        "dependencies": dependencies,
    }
    return descriptor


def _runtime_type_descriptor(
    value: type,
    *,
    context: _RuntimeFingerprintContext,
    path: str,
    depth: int,
    node: int,
) -> dict[str, Any]:
    module_name = _runtime_metadata_text(value, "__module__", path=path)
    qualname = _runtime_metadata_text(value, "__qualname__", path=path)
    if module_name is None or qualname is None:
        raise ValidationError(f"trusted runtime type metadata is invalid at {path}")
    context.consume_text(module_name)
    context.consume_text(qualname)
    descriptor: dict[str, Any] = {
        "kind": "type",
        "node": node,
        "module": module_name,
        "qualname": qualname,
    }
    if module_name != "pyreadr" and not module_name.startswith("pyreadr."):
        descriptor["scope"] = "external"
        return descriptor

    module = sys.modules.get(module_name)
    if type(module) is not types.ModuleType:
        raise ValidationError(f"trusted runtime type module is absent at {path}")
    descriptor["module_binding"] = _runtime_module_descriptor(
        module,
        context=context,
        path=f"{path}.__module_object__",
        depth=depth + 1,
    )
    metaclass = type(value)
    descriptor["metaclass"] = _runtime_execution_descriptor(
        metaclass,
        context=context,
        path=f"{path}.__metaclass__",
        depth=depth + 1,
    )
    bases = type.__getattribute__(value, "__bases__")
    if type(bases) is not tuple or len(bases) > _RUNTIME_FINGERPRINT_MAX_ITEMS:
        raise ValidationError(f"trusted runtime type bases are invalid at {path}")
    descriptor["bases"] = [
        _runtime_execution_descriptor(
            base,
            context=context,
            path=f"{path}.__bases__[{index}]",
            depth=depth + 1,
        )
        for index, base in enumerate(bases)
    ]

    namespace = type.__getattribute__(value, "__dict__")
    if len(namespace) > _RUNTIME_FINGERPRINT_MAX_ITEMS:
        raise ValidationError("trusted runtime type namespace exceeded its item limit")
    members: dict[str, Any] = {}
    for name in sorted(namespace):
        if type(name) is not str:
            raise ValidationError(f"trusted runtime type member is invalid at {path}")
        context.consume_text(name)
        members[name] = _runtime_execution_descriptor(
            namespace[name],
            context=context,
            path=_runtime_child_path(path, "member", name),
            depth=depth + 1,
        )
    descriptor["namespace"] = members
    return descriptor


def _runtime_execution_descriptor(
    value: Any,
    *,
    context: _RuntimeFingerprintContext,
    path: str,
    depth: int,
) -> Any:
    expected = (
        context.expected_bindings.get(path, _RUNTIME_MISSING)
        if context.expected_bindings is not None
        else _RUNTIME_MISSING
    )
    if expected is not _RUNTIME_MISSING and expected is not value:
        raise ValidationError(f"trusted runtime object binding changed at {path}")

    value_type = type(value)
    if value is None or value is Ellipsis or value is NotImplemented:
        context.touch(path, depth)
        return {"kind": value_type.__name__}
    if value_type is bool:
        context.touch(path, depth)
        return {"kind": "bool", "value": value}
    if value_type is int:
        context.touch(path, depth)
        if value.bit_length() > 4_096:
            raise ValidationError("trusted runtime integer state is too large")
        return {"kind": "int", "value": value}
    if value_type is float:
        context.touch(path, depth)
        return {"kind": "float", "hex": value.hex()}
    if value_type is complex:
        context.touch(path, depth)
        return {"kind": "complex", "real": value.real.hex(), "imag": value.imag.hex()}
    if value_type is str:
        context.touch(path, depth)
        context.consume_text(value)
        return {"kind": "str", "value": value}
    if value_type is bytes:
        context.touch(path, depth)
        context.byte_count += len(value)
        if context.byte_count > _RUNTIME_FINGERPRINT_MAX_BYTES:
            raise ValidationError("trusted runtime byte state is too large")
        return {"kind": "bytes", "sha256": hashlib.sha256(value).hexdigest()}
    if value_type is types.CodeType:
        _runtime_guard_code(value, context=context, path=path, depth=depth)
        return {"kind": "code", "value": _trusted_code_descriptor(value)}

    node, repeated = context.bind(path, value, depth)
    if repeated:
        return {"kind": "reference", "node": node}
    if value_type is tuple or value_type is list:
        if len(value) > _RUNTIME_FINGERPRINT_MAX_ITEMS:
            raise ValidationError("trusted runtime sequence exceeded its item limit")
        return {
            "kind": value_type.__name__,
            "node": node,
            "items": [
                _runtime_execution_descriptor(
                    item,
                    context=context,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                )
                for index, item in enumerate(value)
            ],
        }
    if value_type is dict:
        if len(value) > _RUNTIME_FINGERPRINT_MAX_ITEMS or not all(
            type(key) is str for key in value
        ):
            raise ValidationError(f"trusted runtime mapping is invalid at {path}")
        items: dict[str, Any] = {}
        for key in sorted(value):
            context.consume_text(key)
            items[key] = _runtime_execution_descriptor(
                value[key],
                context=context,
                path=_runtime_child_path(path, "mapping", key),
                depth=depth + 1,
            )
        return {"kind": "mapping", "node": node, "items": items}
    if value_type is set or value_type is frozenset:
        if len(value) > _RUNTIME_FINGERPRINT_MAX_ITEMS or not all(
            type(item) in {bool, int, float, str} for item in value
        ):
            raise ValidationError(f"trusted runtime set is invalid at {path}")
        members = [
            _runtime_execution_descriptor(
                item,
                context=context,
                path=f"{path}.member[{index}]",
                depth=depth + 1,
            )
            for index, item in enumerate(
                sorted(
                    value,
                    key=lambda item: _canonical_json_bytes([type(item).__name__, item]),
                )
            )
        ]
        return {"kind": value_type.__name__, "node": node, "items": members}
    if value_type is types.ModuleType:
        return _runtime_module_descriptor(
            value, context=context, path=path, depth=depth, node=node
        )
    if isinstance(value, type):
        return _runtime_type_descriptor(
            value, context=context, path=path, depth=depth, node=node
        )
    if value_type is staticmethod or value_type is classmethod:
        function = object.__getattribute__(value, "__func__")
        return {
            "kind": value_type.__name__,
            "node": node,
            "function": _runtime_execution_descriptor(
                function,
                context=context,
                path=f"{path}.__func__",
                depth=depth + 1,
            ),
        }
    if value_type is property:
        return {
            "kind": "property",
            "node": node,
            "fget": _runtime_execution_descriptor(
                object.__getattribute__(value, "fget"),
                context=context,
                path=f"{path}.fget",
                depth=depth + 1,
            ),
            "fset": _runtime_execution_descriptor(
                object.__getattribute__(value, "fset"),
                context=context,
                path=f"{path}.fset",
                depth=depth + 1,
            ),
            "fdel": _runtime_execution_descriptor(
                object.__getattribute__(value, "fdel"),
                context=context,
                path=f"{path}.fdel",
                depth=depth + 1,
            ),
            "doc": _runtime_execution_descriptor(
                object.__getattribute__(value, "__doc__"),
                context=context,
                path=f"{path}.__doc__",
                depth=depth + 1,
            ),
        }

    code = _runtime_direct_attribute(value, "__code__")
    namespace = _runtime_direct_attribute(value, "__globals__")
    if type(code) is types.CodeType and (
        namespace is _RUNTIME_MISSING or type(namespace) is dict
    ):
        return _runtime_executable_descriptor(
            value,
            code=code,
            namespace=None if namespace is _RUNTIME_MISSING else namespace,
            context=context,
            path=path,
            depth=depth,
            node=node,
        )

    descriptor: dict[str, Any] = {
        "kind": "opaque",
        "node": node,
        "type_module": value_type.__module__,
        "type_qualname": value_type.__qualname__,
        "callable": callable(value),
    }
    instance_state = _runtime_direct_attribute(value, "__dict__")
    if type(instance_state) is dict:
        descriptor["state"] = _runtime_execution_descriptor(
            instance_state,
            context=context,
            path=f"{path}.__dict__",
            depth=depth + 1,
        )
    if value_type is not type:
        descriptor["type"] = _runtime_execution_descriptor(
            value_type,
            context=context,
            path=f"{path}.__type__",
            depth=depth + 1,
        )
    for attribute in ("__module__", "__qualname__", "__name__"):
        metadata = _runtime_direct_attribute(value, attribute)
        if metadata is not _RUNTIME_MISSING and metadata is not None:
            if type(metadata) is not str:
                raise ValidationError(f"trusted runtime metadata is invalid at {path}")
            context.consume_text(metadata)
            descriptor[attribute] = metadata
    return descriptor


def _runtime_executable_fingerprint(
    value: Any,
    *,
    label: str,
    expected_bindings: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if not callable(value):
        raise ValidationError(f"trusted runtime executable is invalid: {label}")
    context = _RuntimeFingerprintContext(
        expected_bindings=expected_bindings, seen={}, bindings={}
    )
    payload = _runtime_execution_descriptor(value, context=context, path=label, depth=0)
    if expected_bindings is not None and set(context.bindings) != set(
        expected_bindings
    ):
        raise ValidationError(
            f"trusted runtime execution graph changed after loading: {label}"
        )
    encoded = _canonical_json_bytes(payload)
    if len(encoded) > _RUNTIME_FINGERPRINT_MAX_BYTES:
        raise ValidationError("trusted runtime fingerprint exceeded its encoded limit")
    return hashlib.sha256(encoded).hexdigest(), context.bindings


def _runtime_parser_class_fingerprint(value: Any, *, label: str) -> str:
    if not isinstance(value, type):
        raise ValidationError(f"trusted runtime parser class is invalid: {label}")
    context = _RuntimeFingerprintContext(expected_bindings=None, seen={}, bindings={})
    payload = _runtime_execution_descriptor(value, context=context, path=label, depth=0)
    encoded = _canonical_json_bytes(payload)
    if len(encoded) > _RUNTIME_FINGERPRINT_MAX_BYTES:
        raise ValidationError("trusted runtime fingerprint exceeded its encoded limit")
    return hashlib.sha256(encoded).hexdigest()


def _seal_runtime_executables(module: Any, *, name: str) -> None:
    sealed: dict[str, tuple[Any, str, dict[str, Any]]] = {}
    for attribute in _RUNTIME_EXECUTABLE_NAMES.get(name, ()):
        value = getattr(module, attribute, None)
        fingerprint, bindings = _runtime_executable_fingerprint(
            value, label=f"{name}.{attribute}"
        )
        sealed[attribute] = (
            value,
            fingerprint,
            bindings,
        )
    if sealed:
        _TRUSTED_RUNTIME_EXECUTABLES[name] = sealed


def _assert_runtime_executables(module: Any, *, name: str) -> None:
    expected = _TRUSTED_RUNTIME_EXECUTABLES.get(name)
    if name in _RUNTIME_EXECUTABLE_NAMES and expected is None:
        raise ValidationError(
            f"runtime executable namespace was not registered by the trusted loader: {name}"
        )
    for attribute, (expected_value, expected_fingerprint, expected_bindings) in (
        expected or {}
    ).items():
        observed = getattr(module, attribute, None)
        if observed is not expected_value:
            raise ValidationError(
                f"trusted runtime executable changed after loading: {name}.{attribute}"
            )
        try:
            observed_fingerprint, _ = _runtime_executable_fingerprint(
                observed,
                label=f"{name}.{attribute}",
                expected_bindings=expected_bindings,
            )
        except ValidationError as exc:
            raise ValidationError(
                "trusted runtime executable global dependency changed after loading: "
                f"{name}.{attribute}"
            ) from exc
        if observed_fingerprint != expected_fingerprint:
            raise ValidationError(
                "trusted runtime executable global dependency changed after loading: "
                f"{name}.{attribute}"
            )


def _install_runtime_sys_module(name: str, module: Any) -> None:
    sys.modules[name] = module


def _bind_trusted_runtime_module(name: str, module: Any) -> None:
    _TRUSTED_RUNTIME_MODULES[name] = module


def _restore_trusted_runtime_modules(
    original_object: dict[str, Any], snapshot: Mapping[str, Any]
) -> None:
    global _TRUSTED_RUNTIME_MODULES
    _restore_mapping_state(original_object, snapshot)
    _TRUSTED_RUNTIME_MODULES = original_object


def _restore_trusted_runtime_executables(
    original_object: dict[str, dict[str, tuple[Any, str, dict[str, Any]]]],
    snapshot: Mapping[str, Any],
) -> None:
    global _TRUSTED_RUNTIME_EXECUTABLES
    _restore_mapping_state(original_object, snapshot)
    _TRUSTED_RUNTIME_EXECUTABLES = original_object


def _load_trusted_runtime_module(module_name: str) -> Any:
    """Execute a scientific package only from its verified distribution origin."""
    with _PROCESS_GLOBAL_STATE_LOCK:
        spec, origin = _trusted_runtime_spec(module_name)
        bound = _TRUSTED_RUNTIME_MODULES.get(module_name)
        existing = sys.modules.get(module_name)
        if bound is not None:
            if existing is not bound:
                raise ValidationError(
                    f"trusted runtime module was replaced after loading: {module_name}"
                )
            _assert_runtime_module_origin(bound, name=module_name, origin=origin)
            _assert_runtime_executables(bound, name=module_name)
            return bound
        if existing is not None:
            _assert_runtime_module_origin(existing, name=module_name, origin=origin)
            if module_name in _RUNTIME_EXECUTABLE_NAMES:
                raise ValidationError(
                    "refusing runtime executable module not registered by the trusted "
                    f"loader: {module_name}"
                )
        original_modules_object = sys.modules
        modules_snapshot = dict(original_modules_object)
        original_runtime_modules = _TRUSTED_RUNTIME_MODULES
        runtime_modules_snapshot = dict(original_runtime_modules)
        original_runtime_executables = _TRUSTED_RUNTIME_EXECUTABLES
        runtime_executables_snapshot = dict(original_runtime_executables)
        loaded_module: Any | None = None
        primary_failure: BaseException | None = None
        cleanup_failures: list[CleanupFailure] = []
        try:
            if existing is not None:
                _bind_trusted_runtime_module(module_name, existing)
                loaded_module = existing
            else:
                parent_name = module_name.rpartition(".")[0]
                if parent_name:
                    _load_trusted_runtime_module(parent_name)
                    spec, origin = _trusted_runtime_spec(module_name)
                module = importlib.util.module_from_spec(spec)
                _install_runtime_sys_module(module_name, module)
                with _trusted_runtime_import_scope():
                    spec.loader.exec_module(module)
                _assert_runtime_module_origin(module, name=module_name, origin=origin)
                _seal_runtime_executables(module, name=module_name)
                _bind_trusted_runtime_module(module_name, module)
                loaded_module = module
        except BaseException as exc:
            primary_failure = exc

        if primary_failure is not None:
            restoration_steps: tuple[tuple[str, Callable[[], None]], ...] = (
                (
                    "trusted runtime executable registry restoration",
                    lambda: _restore_trusted_runtime_executables(
                        original_runtime_executables, runtime_executables_snapshot
                    ),
                ),
                (
                    "trusted runtime module registry restoration",
                    lambda: _restore_trusted_runtime_modules(
                        original_runtime_modules, runtime_modules_snapshot
                    ),
                ),
                (
                    "runtime loader sys.modules restoration",
                    lambda: _restore_runtime_sys_modules(
                        original_modules_object, modules_snapshot
                    ),
                ),
            )
            for attempt in ("initial", "second-chance"):
                for label, cleanup in restoration_steps:
                    _capture_cleanup_failure(
                        cleanup_failures,
                        f"{attempt} {label}",
                        cleanup,
                    )
            _raise_after_cleanup(primary_failure, cleanup_failures)

        if loaded_module is None:
            raise RuntimeError("trusted runtime loader completed without a module")
        return loaded_module


def _runtime_dependency_preflight() -> dict[str, str]:
    """Validate versions and import origins without executing package code."""
    packages: dict[str, str] = {}
    for name in REQUIRED_RUNTIME_DISTRIBUTIONS:
        distribution = _trusted_distribution(name)
        packages[name] = str(distribution.version)
    if packages["pyreadr"] != PYREADR_REQUIRED_VERSION:
        raise ValidationError(
            "pyreadr version differs from the fixed runtime: "
            f"expected {PYREADR_REQUIRED_VERSION}, observed {packages['pyreadr']}"
        )
    for module_name in RUNTIME_MODULE_DISTRIBUTIONS:
        _, origin = _trusted_runtime_spec(module_name)
        bound = _TRUSTED_RUNTIME_MODULES.get(module_name)
        existing = sys.modules.get(module_name)
        if existing is not None:
            _assert_runtime_module_origin(existing, name=module_name, origin=origin)
            if bound is None and module_name in _RUNTIME_EXECUTABLE_NAMES:
                raise ValidationError(
                    "refusing runtime executable module not registered by the trusted "
                    f"loader: {module_name}"
                )
            if bound is not None:
                if existing is not bound:
                    raise ValidationError(
                        f"trusted runtime module was replaced after loading: {module_name}"
                    )
                _assert_runtime_executables(bound, name=module_name)
    return packages


def _load_scientific_runtime() -> dict[str, str]:
    """Validate every dependency origin, then execute the allowlisted modules."""
    packages = _runtime_dependency_preflight()
    for module_name in RUNTIME_MODULE_DISTRIBUTIONS:
        _load_trusted_runtime_module(module_name)
    loaded_pyreadr = _load_trusted_runtime_module("pyreadr")
    if str(getattr(loaded_pyreadr, "__version__", "")) != PYREADR_REQUIRED_VERSION:
        raise ValidationError("pyreadr module and distribution versions disagree")
    if not callable(getattr(loaded_pyreadr, "read_r", None)):
        raise ValidationError("the fixed pyreadr runtime lacks callable read_r")
    return packages


def _environment_inventory() -> dict[str, Any]:
    packages = _runtime_dependency_preflight()
    installed_distributions = sorted(
        (
            {
                "name": str(distribution.metadata.get("Name") or distribution.name),
                "version": str(distribution.version),
            }
            for distribution in importlib.metadata.distributions()
        ),
        key=lambda row: (row["name"].casefold(), row["name"], row["version"]),
    )
    return {
        "python": platform.python_version(),
        "python_full": sys.version,
        "python_build": list(platform.python_build()),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "installed_distributions": installed_distributions,
    }


def _cuda_driver_version() -> str:
    """Return the one installed NVIDIA driver version or fail closed."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValidationError(
            "CUDA driver version cannot be queried with nvidia-smi"
        ) from exc
    versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if (
        not versions
        or len(set(versions)) != 1
        or CUDA_DRIVER_VERSION_RE.fullmatch(versions[0]) is None
    ):
        raise ValidationError("nvidia-smi returned an ambiguous CUDA driver version")
    return versions[0]


def _training_runtime_record(torch: Any, device: str) -> dict[str, Any]:
    """Capture the exact training runtime after deterministic setup has run."""
    if device not in {"cpu", "cuda"}:
        raise ValidationError(f"unsupported resolved training device: {device}")
    packages = {
        name: importlib.metadata.version(name)
        for name in REQUIRED_RUNTIME_DISTRIBUTIONS
    }
    cuda_available = bool(torch.cuda.is_available())
    if device == "cuda":
        if not cuda_available:
            raise ValidationError(
                "CUDA runtime metadata requested while CUDA is unavailable"
            )
        device_index = int(torch.cuda.current_device())
        capability = [
            int(value) for value in torch.cuda.get_device_capability(device_index)
        ]
        cuda_runtime_version = str(torch.version.cuda or "")
        if not cuda_runtime_version:
            raise ValidationError("PyTorch does not report its CUDA runtime version")
        runtime = {
            "cuda_runtime_version": cuda_runtime_version,
            "cuda_driver_version": _cuda_driver_version(),
            "cuda_device_index": device_index,
            "cuda_device_name": str(torch.cuda.get_device_name(device_index)),
            "cuda_device_capability": capability,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
    else:
        runtime = {
            "cuda_runtime_version": None,
            "cuda_driver_version": None,
            "cuda_device_index": None,
            "cuda_device_name": None,
            "cuda_device_capability": None,
            "cudnn_version": None,
            "cudnn_benchmark": None,
            "cudnn_deterministic": None,
            "cuda_matmul_allow_tf32": None,
            "cudnn_allow_tf32": None,
            "cublas_workspace_config": None,
        }
    record = {
        "schema": TRAINING_RUNTIME_SCHEMA,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package_versions": packages,
        "torch_module_version": str(torch.__version__),
        "device_type": device,
        "cuda_available": cuda_available,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        **runtime,
    }
    _validate_training_runtime(record, expected_device=device)
    return record


def _validate_training_runtime(
    runtime: Any,
    *,
    expected_device: str,
) -> None:
    """Validate the closed runtime schema stored in a prediction lock."""
    if not isinstance(runtime, Mapping):
        raise ValidationError("full-refit runtime metadata is not an object")
    keys = set(runtime)
    if keys != TRAINING_RUNTIME_FIELDS:
        missing = sorted(TRAINING_RUNTIME_FIELDS - keys)
        extra = sorted(keys - TRAINING_RUNTIME_FIELDS)
        raise ValidationError(
            f"full-refit runtime metadata fields differ: missing={missing}, extra={extra}"
        )
    for field in (
        "python_version",
        "python_implementation",
        "platform",
        "machine",
        "torch_module_version",
    ):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise ValidationError(f"full-refit runtime field is invalid: {field}")
    if runtime["schema"] != TRAINING_RUNTIME_SCHEMA:
        raise ValidationError("unexpected full-refit runtime schema")
    packages = runtime["package_versions"]
    if (
        not isinstance(packages, Mapping)
        or set(packages) != set(REQUIRED_RUNTIME_DISTRIBUTIONS)
        or any(not isinstance(value, str) or not value for value in packages.values())
    ):
        raise ValidationError("full-refit runtime package versions are incomplete")
    if runtime["device_type"] != expected_device or expected_device not in {
        "cpu",
        "cuda",
    }:
        raise ValidationError("full-refit runtime device does not match training")
    if not isinstance(runtime["cuda_available"], bool):
        raise ValidationError("full-refit CUDA availability flag is invalid")
    if runtime["deterministic_algorithms_enabled"] is not True:
        raise ValidationError(
            "deterministic algorithms were not enabled for full refit"
        )
    debug_mode = runtime["deterministic_debug_mode"]
    if isinstance(debug_mode, bool) or debug_mode != 2:
        raise ValidationError("full-refit deterministic debug mode was not error mode")

    cuda_fields = (
        "cuda_runtime_version",
        "cuda_driver_version",
        "cuda_device_index",
        "cuda_device_name",
        "cuda_device_capability",
        "cudnn_version",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cublas_workspace_config",
    )
    if expected_device == "cpu":
        if any(runtime[field] is not None for field in cuda_fields):
            raise ValidationError("CPU full-refit runtime must use explicit CUDA nulls")
        return

    if runtime["cuda_available"] is not True:
        raise ValidationError("CUDA full-refit runtime reports CUDA unavailable")
    if (
        not isinstance(runtime["cuda_runtime_version"], str)
        or not runtime["cuda_runtime_version"]
        or not isinstance(runtime["cuda_driver_version"], str)
        or CUDA_DRIVER_VERSION_RE.fullmatch(runtime["cuda_driver_version"]) is None
        or not isinstance(runtime["cuda_device_name"], str)
        or not runtime["cuda_device_name"]
    ):
        raise ValidationError("CUDA runtime, driver, or device identity is invalid")
    device_index = runtime["cuda_device_index"]
    if (
        isinstance(device_index, bool)
        or not isinstance(device_index, int)
        or device_index < 0
    ):
        raise ValidationError("CUDA device index is invalid")
    capability = runtime["cuda_device_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in capability
        )
        or capability[0] < 1
        or capability[1] < 0
    ):
        raise ValidationError("CUDA device capability is invalid")
    cudnn_version = runtime["cudnn_version"]
    if (
        isinstance(cudnn_version, bool)
        or not isinstance(cudnn_version, int)
        or cudnn_version <= 0
    ):
        raise ValidationError("cuDNN version is invalid")
    if runtime["cudnn_benchmark"] is not False:
        raise ValidationError(
            "cuDNN benchmark must be disabled for deterministic refit"
        )
    if runtime["cudnn_deterministic"] is not True:
        raise ValidationError("cuDNN deterministic mode must be enabled for full refit")
    for field in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32"):
        if not isinstance(runtime[field], bool):
            raise ValidationError(f"full-refit runtime field is invalid: {field}")
    if runtime["cublas_workspace_config"] not in {":16:8", ":4096:8"}:
        raise ValidationError("CUBLAS_WORKSPACE_CONFIG is not a deterministic setting")


def _trusted_module_origin(module: Any, *, name: str) -> Path:
    reported = getattr(module, "__file__", None)
    if not isinstance(reported, (str, os.PathLike)):
        raise ValidationError(f"trusted local module has no valid origin: {name}")
    try:
        return Path(reported).resolve(strict=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ValidationError(
            f"trusted local module origin cannot be resolved: {name}"
        ) from exc


def _trusted_code_constant(value: Any) -> Any:
    """Return a deterministic descriptor for a Python code-object constant."""
    if value is None or value is Ellipsis or value is NotImplemented:
        return [type(value).__name__]
    if isinstance(value, (bool, int, str)):
        return [type(value).__name__, value]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, complex):
        return ["complex", value.real.hex(), value.imag.hex()]
    if isinstance(value, bytes):
        return ["bytes", len(value), hashlib.sha256(value).hexdigest()]
    if isinstance(value, tuple):
        return ["tuple", [_trusted_code_constant(item) for item in value]]
    if isinstance(value, frozenset):
        members = [_trusted_code_constant(item) for item in value]
        members.sort(key=_canonical_json_bytes)
        return ["frozenset", members]
    if isinstance(value, types.CodeType):
        return ["code", _trusted_code_descriptor(value)]
    raise ValidationError(
        "trusted local callable contains an unsupported code constant: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _trusted_code_descriptor(code: types.CodeType) -> dict[str, Any]:
    """Describe executable code without CPython's mutable quickening state."""
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code_sha256": hashlib.sha256(code.co_code).hexdigest(),
        "constants": [_trusted_code_constant(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "filename": code.co_filename,
        "name": code.co_name,
        "qualname": code.co_qualname,
        "firstlineno": code.co_firstlineno,
        "linetable_sha256": hashlib.sha256(code.co_linetable).hexdigest(),
        "exceptiontable_sha256": hashlib.sha256(code.co_exceptiontable).hexdigest(),
    }


def _trusted_value_fingerprint(
    value: Any,
    *,
    owner_module: str,
    seen: dict[int, int],
) -> Any:
    """Describe execution-relevant local state without invoking user repr methods."""
    if value is None or isinstance(value, (bool, int, str)):
        return [type(value).__name__, value]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, bytes):
        return ["bytes", len(value), hashlib.sha256(value).hexdigest()]
    if isinstance(value, Path):
        return ["path", type(value).__name__, str(value)]
    if isinstance(value, re.Pattern):
        return ["regex", value.pattern, value.flags]
    if isinstance(value, np.generic):
        return ["numpy-scalar", str(value.dtype), value.item()]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            payload = [
                _trusted_value_fingerprint(item, owner_module=owner_module, seen=seen)
                for item in value.reshape(-1).tolist()
            ]
        else:
            payload = hashlib.sha256(
                np.ascontiguousarray(value).view(np.uint8).tobytes()
            ).hexdigest()
        return [
            "numpy-array",
            id(value),
            str(value.dtype),
            list(value.shape),
            list(value.strides),
            bool(value.flags.writeable),
            payload,
        ]

    object_id = id(value)
    if object_id in seen:
        return ["reference", seen[object_id]]
    seen[object_id] = len(seen)

    if isinstance(value, types.FunctionType):
        closure: list[Any] = []
        for cell in value.__closure__ or ():
            try:
                cell_value = cell.cell_contents
            except ValueError:
                closure.append(["empty-cell"])
            else:
                closure.append(
                    _trusted_value_fingerprint(
                        cell_value, owner_module=owner_module, seen=seen
                    )
                )
        return [
            "python-function",
            object_id,
            value.__module__,
            value.__qualname__,
            _trusted_code_descriptor(value.__code__),
            _trusted_value_fingerprint(
                value.__defaults__, owner_module=owner_module, seen=seen
            ),
            _trusted_value_fingerprint(
                value.__kwdefaults__, owner_module=owner_module, seen=seen
            ),
            _trusted_value_fingerprint(
                value.__annotations__, owner_module=owner_module, seen=seen
            ),
            _trusted_value_fingerprint(
                value.__dict__, owner_module=owner_module, seen=seen
            ),
            closure,
        ]
    if isinstance(value, staticmethod):
        return [
            "staticmethod",
            _trusted_value_fingerprint(
                value.__func__, owner_module=owner_module, seen=seen
            ),
        ]
    if isinstance(value, classmethod):
        return [
            "classmethod",
            _trusted_value_fingerprint(
                value.__func__, owner_module=owner_module, seen=seen
            ),
        ]
    if isinstance(value, property):
        return [
            "property",
            _trusted_value_fingerprint(
                value.fget, owner_module=owner_module, seen=seen
            ),
            _trusted_value_fingerprint(
                value.fset, owner_module=owner_module, seen=seen
            ),
            _trusted_value_fingerprint(
                value.fdel, owner_module=owner_module, seen=seen
            ),
        ]
    if isinstance(value, type):
        if value.__module__ != owner_module:
            return [
                "external-type",
                object_id,
                value.__module__,
                value.__qualname__,
            ]
        namespace = [
            [
                key,
                _trusted_value_fingerprint(item, owner_module=owner_module, seen=seen),
            ]
            for key, item in value.__dict__.items()
            if key not in {"__dict__", "__weakref__"}
        ]
        return [
            "local-type",
            object_id,
            value.__module__,
            value.__qualname__,
            namespace,
        ]
    if isinstance(value, types.ModuleType):
        origin = getattr(value, "__file__", None)
        return ["module", object_id, value.__name__, str(origin) if origin else None]
    if isinstance(value, Mapping):
        return [
            "mapping",
            object_id,
            [
                [
                    _trusted_value_fingerprint(
                        key, owner_module=owner_module, seen=seen
                    ),
                    _trusted_value_fingerprint(
                        item, owner_module=owner_module, seen=seen
                    ),
                ]
                for key, item in value.items()
            ],
        ]
    if isinstance(value, (list, tuple)):
        return [
            type(value).__name__,
            object_id,
            [
                _trusted_value_fingerprint(item, owner_module=owner_module, seen=seen)
                for item in value
            ],
        ]
    if isinstance(value, (set, frozenset)):
        members = [
            _trusted_value_fingerprint(
                item, owner_module=owner_module, seen=seen.copy()
            )
            for item in value
        ]
        members.sort(key=_canonical_json_bytes)
        return [type(value).__name__, object_id, members]
    return [
        "opaque-object",
        object_id,
        type(value).__module__,
        type(value).__qualname__,
    ]


def _trusted_module_namespace_fingerprint(module: Any, *, name: str) -> str:
    """Seal local callables and mutable module globals for same-process reuse."""
    namespace = [
        [
            key,
            _trusted_value_fingerprint(value, owner_module=name, seen={}),
        ]
        for key, value in module.__dict__.items()
        if key not in {"__builtins__", "__warningregistry__"}
    ]
    return hashlib.sha256(_canonical_json_bytes(namespace)).hexdigest()


def _assert_trusted_module_namespace(
    module: Any, *, name: str, expected_fingerprint: str
) -> None:
    observed = _trusted_module_namespace_fingerprint(module, name=name)
    if observed != expected_fingerprint:
        raise ValidationError(
            "trusted local module namespace or callable fingerprint changed "
            f"after execution: {name}"
        )


def _load_trusted_module(
    name: str,
    *,
    expected_source_record: Mapping[str, Any] | None = None,
    root: Path = ROOT,
) -> Any:
    """Execute or reuse one local dependency under the process-state lock."""
    with _PROCESS_GLOBAL_STATE_LOCK:
        return _load_trusted_module_locked(
            name,
            expected_source_record=expected_source_record,
            root=root,
        )


def _load_trusted_module_locked(
    name: str,
    *,
    expected_source_record: Mapping[str, Any] | None = None,
    root: Path = ROOT,
) -> Any:
    """Execute one local dependency from hash-bound bytes exactly once.

    A same-path object already present in ``sys.modules`` is not evidence that
    the current source bytes produced that object.  Only modules created here
    from a stable read are reusable, and reuse requires the original object
    identity and unchanged source record.
    """
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
        raise ValidationError(f"unsafe local module name: {name!r}")
    expected = (root / "code" / f"{name}.py").resolve(strict=True)
    source_bytes, source_record = _stable_read_bytes(
        expected,
        label=f"trusted local module source {name}",
        root=root,
    )
    if expected_source_record is not None:
        _assert_same_record(
            source_record,
            expected_source_record,
            label=f"frozen trusted local module source {name}",
        )
    binding = _TRUSTED_LOCAL_MODULES.get(name)
    existing = sys.modules.get(name)
    if binding is not None:
        trusted_module, trusted_record, trusted_fingerprint = binding
        if existing is not trusted_module:
            raise ValidationError(
                f"refusing replaced or stale sys.modules entry for trusted module: {name}"
            )
        _assert_same_record(
            source_record,
            trusted_record,
            label=f"trusted local module source {name}",
        )
        origin = _trusted_module_origin(trusted_module, name=name)
        if origin != expected:
            raise ValidationError(
                f"trusted local module origin changed after execution: {name}={origin}"
            )
        _assert_trusted_module_namespace(
            trusted_module,
            name=name,
            expected_fingerprint=trusted_fingerprint,
        )
        return trusted_module
    if existing is not None:
        origin = _trusted_module_origin(existing, name=name)
        raise ValidationError(
            "refusing preloaded local module without execution provenance: "
            f"{name}={origin}"
        )

    module = types.ModuleType(name)
    module.__file__ = str(expected)
    module.__package__ = ""
    module.__loader__ = None
    had_module = name in sys.modules
    previous_module = sys.modules.get(name)
    previous_binding = _TRUSTED_LOCAL_MODULES.get(name)
    code_dir = str(expected.parent)
    original_path_object = sys.path
    original_path_values = list(original_path_object)
    primary_failure: BaseException | None = None
    cleanup_failures: list[CleanupFailure] = []

    def rollback_module_installation() -> None:
        if previous_binding is not None:
            _TRUSTED_LOCAL_MODULES[name] = previous_binding
        else:
            _TRUSTED_LOCAL_MODULES.pop(name, None)
        if had_module:
            sys.modules[name] = cast(types.ModuleType, previous_module)
        else:
            sys.modules.pop(name, None)

    try:
        try:
            sys.modules[name] = module
            sys.path.insert(0, code_dir)
            compiled = compile(source_bytes, str(expected), "exec", dont_inherit=True)
            exec(compiled, module.__dict__)
        except BaseException as exc:
            primary_failure = exc
    finally:
        try:
            _capture_cleanup_failure(
                cleanup_failures,
                "initial sys.path restoration",
                lambda: _restore_sys_path(original_path_object, original_path_values),
            )
            if primary_failure is None and not cleanup_failures:
                if sys.modules.get(name) is not module:
                    raise ValidationError(
                        f"trusted local module replaced itself during execution: {name}"
                    )
                origin = _trusted_module_origin(module, name=name)
                if origin != expected:
                    raise ValidationError(
                        f"module path validation failed: {name}={origin}"
                    )
                namespace_fingerprint = _trusted_module_namespace_fingerprint(
                    module, name=name
                )
                _TRUSTED_LOCAL_MODULES[name] = (
                    module,
                    source_record,
                    namespace_fingerprint,
                )
        except BaseException as exc:
            if primary_failure is None:
                primary_failure = exc
            else:
                cleanup_failures.append(
                    ("initial sys.path restoration or module validation", exc)
                )
        try:
            _capture_cleanup_failure(
                cleanup_failures,
                "second-chance sys.path restoration",
                lambda: _restore_sys_path(original_path_object, original_path_values),
            )
        except BaseException as exc:
            if primary_failure is None:
                primary_failure = exc
            else:
                cleanup_failures.append(
                    ("second-chance sys.path restoration transition", exc)
                )
        if primary_failure is not None or cleanup_failures:
            try:
                _capture_cleanup_failure(
                    cleanup_failures,
                    "initial trusted-module installation rollback",
                    rollback_module_installation,
                )
            except BaseException as exc:
                cleanup_failures.append(
                    (
                        "initial trusted-module installation rollback transition",
                        exc,
                    )
                )
            try:
                _capture_cleanup_failure(
                    cleanup_failures,
                    "second-chance trusted-module installation rollback",
                    rollback_module_installation,
                )
            except BaseException as exc:
                cleanup_failures.append(
                    (
                        "second-chance trusted-module installation rollback transition",
                        exc,
                    )
                )
            _raise_after_cleanup(primary_failure, cleanup_failures)
    return module


def _validate_loaded_local_modules(names: Sequence[str], *, root: Path = ROOT) -> None:
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            continue
        binding = _TRUSTED_LOCAL_MODULES.get(name)
        if binding is None or binding[0] is not module:
            raise ValidationError(
                f"refusing local module without trusted execution provenance: {name}"
            )
        _load_trusted_module(name, root=root)


def _frozen_trusted_module_source_records(
    freeze: Mapping[str, Any], names: Sequence[str]
) -> list[dict[str, Any]]:
    """Select the frozen source records for an ordered set of local modules."""
    records = freeze.get("code_and_protocol")
    if not isinstance(records, list):
        raise ValidationError("protocol lock lacks the code/protocol inventory")
    by_path: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        fields = _record_fields(record, f"code_and_protocol[{index}]")
        path = fields["path"]
        if not isinstance(path, str) or path in by_path:
            raise ValidationError(
                "protocol lock has duplicate or malformed code/protocol paths"
            )
        by_path[path] = fields
    selected: list[dict[str, Any]] = []
    for name in names:
        expected_path = f"code/{name}.py"
        record = by_path.get(expected_path)
        if record is None:
            raise ValidationError(
                f"protocol lock lacks trusted local module source: {expected_path}"
            )
        selected.append(dict(record))
    return selected


def _assert_trusted_modules_match_freeze(
    freeze: Mapping[str, Any], names: Sequence[str], *, root: Path = ROOT
) -> list[dict[str, Any]]:
    """Bind exact source bytes executed by trusted modules to the protocol lock."""
    expected_records = _frozen_trusted_module_source_records(freeze, names)
    observed_records: list[dict[str, Any]] = []
    for name, expected_record in zip(names, expected_records, strict=True):
        module = _load_trusted_module(
            name, expected_source_record=expected_record, root=root
        )
        binding = _TRUSTED_LOCAL_MODULES.get(name)
        if binding is None or binding[0] is not module:
            raise ValidationError(
                f"trusted local module lost execution provenance: {name}"
            )
        observed_record = _record_fields(
            binding[1], f"executed trusted local module source {name}"
        )
        _assert_same_record(
            observed_record,
            expected_record,
            label=f"executed trusted local module source {name}",
        )
        observed_records.append(observed_record)
    return observed_records


def audit_features(
    *,
    features_dir: Path = DEFAULT_FEATURE_DIR,
    genes_path: Path = DEFAULT_GENES,
    output: Path | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Validate non-outcome features without accepting or probing an outcome path."""
    _runtime_dependency_preflight()
    paths = {name: features_dir / filename for name, filename in FEATURE_FILES.items()}
    headers: dict[str, list[str]] = {}
    feature_bytes: dict[str, bytes] = {}
    feature_records: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        payload, record = _stable_read_bytes(
            path, label=f"{label} feature input", root=root
        )
        feature_bytes[label] = payload
        feature_records[label] = record
        if label != "figshare_metadata":
            headers[label] = _csv_header_bytes(payload, label)
    metadata = _load_figshare_metadata_bytes(feature_bytes["figshare_metadata"])
    feature_remote: dict[str, Any] = {}
    for label, expected in FEATURE_INPUTS.items():
        remote = _figshare_file(metadata, int(expected["id"]))
        for field in ("name", "bytes", "md5"):
            if remote[field] != expected[field]:
                raise ValidationError(
                    f"publisher feature metadata drift for {label}.{field}"
                )
        local = paths[label]
        if (
            local.name != expected["name"]
            or feature_records[label]["bytes"] != expected["bytes"]
        ):
            raise ValidationError(
                f"local {label} does not match pinned Figshare v1 name/size"
            )
        if (
            hashlib.md5(feature_bytes[label], usedforsecurity=False).hexdigest()
            != expected["md5"]
        ):
            raise ValidationError(
                f"local {label} does not match pinned Figshare v1 MD5"
            )
        feature_remote[label] = remote

    covariates = pd.read_csv(
        io.BytesIO(feature_bytes["covariates"]), dtype={"sample_ID": str}
    )
    if list(covariates.columns) != [
        "sample_ID",
        "ploidy",
        "msStatus",
        "library",
        "tissue",
    ]:
        raise ValidationError("covariate columns are not canonical")
    sample_ids = _require_unique(covariates["sample_ID"], "covariate sample_ID")
    if set(covariates["tissue"].astype(str)) - set(LINEAGE_MAP):
        raise ValidationError("covariates contain a tissue outside the fixed mapping")
    _finite_numeric(covariates[["ploidy", "msStatus"]], "modeled covariates")
    try:
        library_values = pd.to_numeric(covariates["library"], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "unused library metadata contains non-numeric values"
        ) from exc
    if np.isinf(library_values).any():
        raise ValidationError("unused library metadata contains infinite values")
    library_missing = int(np.isnan(library_values).sum())

    if not headers["rna"] or headers["rna"][0] != "sample_ID":
        raise ValidationError("RNA first column must be sample_ID")
    rna = pd.read_csv(io.BytesIO(feature_bytes["rna"]), dtype={"sample_ID": str})
    rna_ids = _require_unique(rna["sample_ID"], "RNA sample_ID")
    rna_columns = [str(column) for column in rna.columns if column != "sample_ID"]
    if not rna_columns or any(not column.endswith("_exp") for column in rna_columns):
        raise ValidationError("RNA columns must be unique *_exp features")
    rna_genes = _require_unique((column[:-4] for column in rna_columns), "RNA gene")
    _finite_numeric(rna[rna_columns], "RNA matrix")

    binary_audits: dict[str, Any] = {}
    for label, suffix in (("driver_mutations", "_mut"), ("driver_variants", None)):
        if (
            not headers[label]
            or headers[label][0] != "sample_ID"
            or len(headers[label]) < 2
        ):
            raise ValidationError(
                f"{label} must start with sample_ID and contain features"
            )
        frame = pd.read_csv(io.BytesIO(feature_bytes[label]), dtype={"sample_ID": str})
        ids = _require_unique(frame["sample_ID"], f"{label} sample_ID")
        if ids != sample_ids:
            raise ValidationError(
                f"{label} sample order does not exactly match covariates"
            )
        columns = [str(column) for column in frame.columns if column != "sample_ID"]
        _require_unique(columns, f"{label} feature")
        if suffix and any(not column.endswith(suffix) for column in columns):
            raise ValidationError(f"{label} columns must end with {suffix}")
        values = _finite_numeric(frame[columns], label)
        if not np.isin(values, (0.0, 1.0)).all():
            raise ValidationError(f"{label} must be binary")
        binary_audits[label] = {
            "samples": len(ids),
            "features": len(columns),
            "positive_entries": int(values.sum()),
        }

    if not set(rna_ids).issubset(sample_ids):
        raise ValidationError("RNA contains sample IDs absent from covariates")
    genes_bytes, genes_record = _stable_read_bytes(
        genes_path,
        label="processed gene manifest",
        root=root,
    )
    processed_genes = _read_lines_bytes(genes_bytes, genes_path)
    overlap = sorted(set(rna_genes) & set(processed_genes))
    missing_rna = sorted(set(sample_ids) - set(rna_ids))
    tissue_counts = {
        str(key): int(value)
        for key, value in sorted(Counter(covariates["tissue"].astype(str)).items())
    }
    gates = {
        "rna_samples_at_least_200": len(rna_ids) >= MIN_RNA_SAMPLES,
        "rna_training_gene_overlap_at_least_10000": len(overlap)
        >= MIN_RNA_GENE_OVERLAP,
        "tissues_exactly_fixed": set(tissue_counts) == set(LINEAGE_MAP),
        "driver_rows_match_covariates": all(
            item["samples"] == len(sample_ids) for item in binary_audits.values()
        ),
    }
    report = {
        "schema": AUDIT_SCHEMA,
        "created_utc": _utc_now(),
        "scientific_result": False,
        "numeric_outcome_loaded_by_this_phase": False,
        "scope": "non-outcome feature feasibility and integrity only; not a chronology attestation",
        "figshare": {"item_id": FIGSHARE_ITEM_ID, "doi": FIGSHARE_DOI, "version": 1},
        "feature_files": feature_records,
        "processed_gene_manifest": genes_record,
        "prediction_axis_bindings": _prediction_axis_bindings(processed_genes, rna_ids),
        "publisher_file_metadata": feature_remote,
        "publisher_outcome_file_metadata": _outcome_metadata(metadata),
        "cohort": {
            "covariate_samples": len(sample_ids),
            "rna_samples": len(rna_ids),
            "rna_missing_sample_ids": missing_rna,
            "tissue_counts": tissue_counts,
            "lineage_mapping": LINEAGE_MAP,
            "model_input_covariates": ["tissue"],
            "unused_covariate_missingness": {"library": library_missing},
            "unused_covariate_policy": "record missingness; do not impute and do not use in prediction",
        },
        "features": {
            "rna_genes": len(rna_genes),
            "rna_training_gene_overlap": len(overlap),
            "rna_training_gene_overlap_sha256": hashlib.sha256(
                "\0".join(overlap).encode("utf-8")
            ).hexdigest(),
            **binary_audits,
            "mutation_covariate_limitation": "Only publisher-provided driver mutation indicators are available externally; this is narrower than Broad somatic mutation input.",
        },
        "gates": gates,
        "audit_passed": all(gates.values()),
    }
    if output is not None:
        _atomic_json(_project_output_path(output, root=root), report, root=root)
    return report


def _reconstruct_prediction_axes(
    freeze: Mapping[str, Any], *, root: Path
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Rebuild the exact model axes from hash-bound source bytes."""
    training_inputs = freeze.get("training_inputs")
    if not isinstance(training_inputs, list):
        raise ValidationError("protocol lock lacks the training input inventory")
    gene_record = next(
        (
            record
            for record in training_inputs
            if isinstance(record, Mapping)
            and record.get("path") == "data/processed/genes.txt"
        ),
        None,
    )
    if gene_record is None:
        raise ValidationError("protocol lock lacks the processed gene manifest")
    gene_path = _verify_record(gene_record, root=root, label="processed genes")
    gene_bytes, consumed_gene_record = _stable_read_bytes(
        gene_path, label="processed genes", root=root
    )
    _assert_same_record(
        consumed_gene_record, gene_record, label="processed gene manifest"
    )
    gene_ids = _read_lines_bytes(gene_bytes, gene_path)

    feature_records = freeze.get("feature_files")
    if not isinstance(feature_records, Mapping):
        raise ValidationError("protocol lock lacks external feature bindings")
    sample_axes: dict[str, list[str]] = {}
    for label in ("covariates", "rna", "driver_mutations"):
        record = feature_records.get(label)
        if not isinstance(record, Mapping):
            raise ValidationError(f"protocol lock lacks frozen {label} input")
        path = _verify_record(record, root=root, label=f"feature_files.{label}")
        payload, consumed_record = _stable_read_bytes(
            path, label=f"frozen {label}", root=root
        )
        _assert_same_record(consumed_record, record, label=f"frozen {label}")
        header = _csv_header_bytes(payload, label)
        if not header or header[0] != "sample_ID":
            raise ValidationError(f"frozen {label} must start with sample_ID")
        frame = pd.read_csv(io.BytesIO(payload), dtype={"sample_ID": str})
        sample_axes[label] = _require_unique(
            frame["sample_ID"], f"frozen {label} sample_ID"
        )
    if sample_axes["driver_mutations"] != sample_axes["covariates"]:
        raise ValidationError(
            "frozen driver-mutation sample order differs from covariates"
        )
    eligible_sample_ids = sample_axes["rna"]
    covariate_ids = set(sample_axes["covariates"])
    mutation_ids = set(sample_axes["driver_mutations"])
    if not set(eligible_sample_ids).issubset(covariate_ids) or not set(
        eligible_sample_ids
    ).issubset(mutation_ids):
        raise ValidationError(
            "frozen RNA cohort contains samples absent from required features"
        )
    bindings = _prediction_axis_bindings(gene_ids, eligible_sample_ids)
    return gene_ids, eligible_sample_ids, bindings


def _validate_campaign(
    path: Path,
    *,
    root: Path = ROOT,
    expected_source_record: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validator = _load_trusted_module(
        "sanger_independent_validation",
        expected_source_record=expected_source_record,
        root=root,
    )
    try:
        return validator.validate_campaign_report(path)
    except Exception as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(f"canonical campaign validation failed: {exc}") from exc


def _artifact_map(run: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = run.get("fold_artifacts")
    if not isinstance(artifacts, list):
        raise ValidationError(f"campaign run lacks fold artifacts: {run.get('key')}")
    output: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValidationError(f"invalid campaign artifact record: {run.get('key')}")
        name = str(artifact["path"])
        if name in output:
            raise ValidationError(
                f"duplicate campaign artifact path: {run.get('key')}:{name}"
            )
        output[name] = artifact
    return output


def _canonical_config(condition: str, config: Mapping[str, Any]) -> dict[str, Any]:
    expected_model = {"DepPLM": "depplm", "C-PLM-MLP": "plmmlp"}[condition]
    required = {"model", "hidden", "lr", "wd", "lamb_sel"}
    if set(config) != required or str(config.get("model")) != expected_model:
        raise ValidationError(f"noncanonical selected configuration for {condition}")
    canonical = {
        "model": expected_model,
        "hidden": int(config["hidden"]),
        "lr": float(config["lr"]),
        "wd": float(config["wd"]),
        "lamb_sel": float(config["lamb_sel"]),
    }
    grid = {(hidden, lr, 1e-5, 0.1) for hidden in (128, 256) for lr in (1e-3, 3e-4)}
    if (
        canonical["hidden"],
        canonical["lr"],
        canonical["wd"],
        canonical["lamb_sel"],
    ) not in grid:
        raise ValidationError(
            f"selected configuration is outside the verified P1 grid: {condition}"
        )
    return canonical


def _config_rank(config: Mapping[str, Any]) -> int:
    order = [(128, 1e-3), (128, 3e-4), (256, 1e-3), (256, 3e-4)]
    return order.index((int(config["hidden"]), float(config["lr"])))


def derive_training_plan(
    report: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    """Derive all full-cohort settings solely from bound P1 fold results."""
    source_runs = report.get("source_runs")
    if not isinstance(source_runs, list):
        raise ValidationError("campaign source_runs are missing")
    indexed = {str(run.get("key")): run for run in source_runs if isinstance(run, dict)}
    plans: list[dict[str, Any]] = []
    outer_prediction_records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            key = f"p1|{condition}|seed-{seed}"
            run = indexed.get(key)
            if run is None:
                raise ValidationError(f"campaign lacks required P1 run: {key}")
            run_path_value = run.get("path")
            if not isinstance(run_path_value, str):
                raise ValidationError(f"campaign run path is invalid: {key}")
            run_path = _safe_project_path(run_path_value, root)
            if not run_path.is_dir():
                raise ValidationError(f"campaign run is not a directory: {key}")
            artifacts = _artifact_map(run)
            configs: list[dict[str, Any]] = []
            epochs: list[int] = []
            fold_sources: list[dict[str, Any]] = []
            for fold_number, fold in enumerate(FOLDS):
                result_name = f"{fold}-result.json"
                prediction_name = f"{fold}-predictions.npz"
                if result_name not in artifacts or prediction_name not in artifacts:
                    raise ValidationError(
                        f"P1 run lacks required fold files: {key}:{fold}"
                    )
                result_path = run_path / result_name
                prediction_path = run_path / prediction_name
                result_bytes, result_record = _stable_read_bytes(
                    result_path,
                    label=f"P1 fold result {key}:{fold}",
                    root=root,
                )
                _assert_same_content(
                    result_record, artifacts[result_name], label=f"{key}:{result_name}"
                )
                result = _parse_json_mapping(
                    result_bytes, f"P1 fold result {key}:{fold}"
                )
                prediction_record = _record_file(prediction_path, root)
                _assert_same_content(
                    prediction_record,
                    artifacts[prediction_name],
                    label=f"{key}:{prediction_name}",
                )
                expected_fields = {
                    "protocol": "p1",
                    "condition": condition,
                    "seed": seed,
                    "fold": fold,
                    "fold_number": fold_number,
                }
                if any(
                    result.get(name) != value for name, value in expected_fields.items()
                ):
                    raise ValidationError(f"P1 fold identity mismatch: {key}:{fold}")
                config = _canonical_config(condition, result.get("best_config", {}))
                epoch = int(result.get("best_epoch", 0))
                if epoch < 1 or epoch > 20:
                    raise ValidationError(
                        f"P1 best_epoch is outside 1..20: {key}:{fold}"
                    )
                configs.append(config)
                epochs.append(epoch)
                fold_sources.append(
                    {
                        "fold": fold,
                        "best_config": config,
                        "best_epoch": epoch,
                        "result": result_record,
                        "outer_prediction": prediction_record,
                    }
                )
                outer_prediction_records.append(prediction_record)
            encoded = [
                _canonical_json_bytes(config).decode("ascii").strip()
                for config in configs
            ]
            counts = Counter(encoded)
            maximum = max(counts.values())
            candidates = [
                configs[index]
                for index, value in enumerate(encoded)
                if counts[value] == maximum
            ]
            selected = min(candidates, key=_config_rank)
            selected_epoch = int(statistics.median(epochs))
            plans.append(
                {
                    "condition": condition,
                    "training_seed": seed,
                    "full_refit_seed_derivation": "derive_seed(training_seed, 'organoid-full-refit', condition)",
                    "selected_config": selected,
                    "selected_epoch": selected_epoch,
                    "config_rule": "modal fold best_config; ties use canonical grid order",
                    "epoch_rule": "median of five P1 fold best_epoch values",
                    "fold_sources": fold_sources,
                }
            )
    return {
        "source_protocol": "p1",
        "conditions": list(CONDITIONS),
        "training_seeds": list(SEEDS),
        "cohort": "all 1103 Broad/DepMap training cell lines",
        "validation_outcome_used_for_selection": False,
        "batch_size": 64,
        "gene_subsample": 6_000,
        "top_mut": 5_000,
        "n_expr_pc": 128,
        "scheduler_horizon": 20,
        "plans": plans,
        "outer_prediction_bindings": outer_prediction_records,
    }


def _embedding_bundle_provenance(
    root: Path = ROOT,
    *,
    expected_source_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    module = _load_trusted_module(
        "embedding_provenance",
        expected_source_record=expected_source_record,
        root=root,
    )
    try:
        provenance = module.validate_embedding_bundle(root)
    except Exception as exc:
        raise ValidationError(f"ESM embedding bundle validation failed: {exc}") from exc
    if not isinstance(provenance, Mapping):
        raise ValidationError("ESM embedding validator returned a non-object result")
    return json.loads(_canonical_json_bytes(provenance))


def _official_source_record(
    repository: Path | None = None, *, root: Path = ROOT
) -> dict[str, Any]:
    repository = repository or (root.parent / "SangerOrganoidBiobank-official")
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError("official Sanger code repository is unavailable") from exc
    if commit != OFFICIAL_REPOSITORY_COMMIT:
        raise ValidationError(
            "official Sanger code checkout is not at the pinned commit"
        )
    relative_files = (
        "scripts/CRISPR_biomarker_analysis.R",
        "scripts/CRISPR_pipeline.R",
    )
    try:
        changed = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--",
                *relative_files,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError(
            "official Sanger source cleanliness cannot be checked"
        ) from exc
    if changed:
        raise ValidationError(
            "official Sanger source files differ from the pinned commit"
        )
    records = []
    for relative in relative_files:
        path = repository / Path(*PurePosixPath(relative).parts)
        digest, size = _stable_hash_and_size(path, "sha256")
        records.append(
            {"repository_relative_path": relative, "bytes": size, "sha256": digest}
        )
    return {
        "repository": "SangerOrganoidBiobank official analysis code",
        "commit": commit,
        "files": records,
        "runtime_dependency": False,
    }


def _outcome_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label, expected in OUTCOME_FILES.items():
        observed = _figshare_file(metadata, int(expected["id"]))
        for field in ("name", "bytes", "md5"):
            if observed[field] != expected[field]:
                raise ValidationError(
                    f"remote outcome metadata drift for {label}.{field}"
                )
        output[label] = observed
    return output


def create_freeze(
    *,
    output: Path = DEFAULT_FREEZE,
    campaign_path: Path = DEFAULT_CAMPAIGN,
    features_dir: Path = DEFAULT_FEATURE_DIR,
    genes_path: Path = DEFAULT_GENES,
    protocol_path: Path = PROTOCOL_PATH,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Create a write-once fixed-protocol revision without an outcome argument."""
    execution_source_records = _execution_source_records()
    _runtime_dependency_preflight()
    canonical_genes = root / "data" / "processed" / "genes.txt"
    if genes_path.resolve(strict=True) != canonical_genes.resolve(strict=True):
        raise ValidationError("gene manifest path is not the canonical training input")
    output = _project_output_path(output, root=root)
    canonical_protocol = (
        root / "docs" / "revision_v2" / "ORGANOID_INDEPENDENT_VALIDATION_PROTOCOL.md"
    )
    canonical_campaign = (
        root
        / "results"
        / "corrected_v2"
        / "campaigns"
        / "20261020T230939.640594900Z-campaign-c41b987c"
        / "campaign.json"
    )
    if protocol_path.resolve(strict=True) != canonical_protocol.resolve(strict=True):
        raise ValidationError("protocol path is not the canonical fixed protocol")
    if campaign_path.resolve(strict=True) != canonical_campaign.resolve(strict=True):
        raise ValidationError("campaign path is not the canonical 99-run campaign")

    for relative in execution_source_records:
        if CODE_DEPENDENCIES.count(relative) != 1:
            raise ValidationError(
                "code/protocol dependency inventory must contain each execution "
                "source exactly once"
            )

    default_protocol_relative = PurePosixPath(
        "docs/revision_v2/ORGANOID_INDEPENDENT_VALIDATION_PROTOCOL.md"
    )
    code_paths = [
        (
            protocol_path
            if PurePosixPath(relative) == default_protocol_relative
            else root / Path(*PurePosixPath(relative).parts)
        )
        for relative in CODE_DEPENDENCIES
    ]
    if len({path.resolve(strict=True) for path in code_paths}) != len(code_paths):
        raise ValidationError(
            "code/protocol dependency inventory contains duplicate files"
        )
    code_records = []
    for relative, path in zip(CODE_DEPENDENCIES, code_paths, strict=True):
        observed_record = _record_file(path, root)
        execution_record = execution_source_records.get(relative)
        if execution_record is not None:
            _assert_same_record(
                observed_record,
                execution_record,
                label=f"execution code dependency {relative}",
            )
            code_records.append(execution_record)
        else:
            code_records.append(observed_record)
    bound_module_names = ("sanger_independent_validation", "embedding_provenance")
    bound_module_records = dict(
        zip(
            bound_module_names,
            _frozen_trusted_module_source_records(
                {"code_and_protocol": code_records}, bound_module_names
            ),
            strict=True,
        )
    )
    audit = audit_features(features_dir=features_dir, genes_path=genes_path, root=root)
    if not audit["audit_passed"]:
        raise ValidationError("non-outcome feature audit did not pass")
    report, consumed_campaign_record = _validate_campaign(
        campaign_path,
        root=root,
        expected_source_record=bound_module_records["sanger_independent_validation"],
    )
    training_plan = derive_training_plan(report, root=root)
    publisher_outcomes = audit["publisher_outcome_file_metadata"]
    input_records = [
        _record_file(root / Path(*PurePosixPath(relative).parts), root)
        for relative in TRAINING_INPUTS
    ]
    embedding_provenance = _embedding_bundle_provenance(
        root,
        expected_source_record=bound_module_records["embedding_provenance"],
    )
    body: dict[str, Any] = {
        "schema": FREEZE_SCHEMA,
        "created_utc": _utc_now(),
        "write_once": True,
        "scientific_result": False,
        "numeric_outcome_argument_accepted_by_this_phase": False,
        "numeric_outcome_loaded_by_this_phase": False,
        "selection_outcome_used": False,
        "registration_status": HISTORICAL_REVISION_PROVENANCE["status"],
        "historical_revision_provenance": HISTORICAL_REVISION_PROVENANCE,
        "required_formal_invocation": _formal_invocation_constraint(),
        "campaign": _record_fields(consumed_campaign_record, "canonical campaign"),
        "training_plan": training_plan,
        "training_inputs": input_records,
        "code_and_protocol": code_records,
        "feature_audit_canonical_sha256": hashlib.sha256(
            _canonical_json_bytes(audit)
        ).hexdigest(),
        "feature_audit": audit,
        "feature_files": audit["feature_files"],
        "prediction_axis_bindings": audit["prediction_axis_bindings"],
        "figshare": {
            "item_id": FIGSHARE_ITEM_ID,
            "doi": FIGSHARE_DOI,
            "version": 1,
            "metadata": dict(audit["feature_files"]["figshare_metadata"]),
            "publisher_outcome_files": publisher_outcomes,
            "primary_outcome_key": "primary_qnorm_lfc",
            "binary_depletion_use": "metadata-bound only; not an endpoint in this protocol",
        },
        "official_analysis_source": _official_source_record(root=root),
        "embedding_bundle_provenance": embedding_provenance,
        "cohort_and_feature_gates": {
            "minimum_eligible_organoids": MIN_PRIMARY_ORGANOIDS,
            "minimum_common_outcome_genes": MIN_PRIMARY_GENES,
            "minimum_genes_per_organoid": MIN_GENES_PER_ORGANOID,
            "minimum_rna_training_gene_overlap": MIN_RNA_GENE_OVERLAP,
            "allowed_tissues": list(LINEAGE_MAP),
            "tissue_to_broad_lineage": LINEAGE_MAP,
            "rna_required_for_prediction": True,
            "nonfinite_model_inputs_predictions_or_external_outcomes_allowed": False,
            "broad_training_target_policy": "NaN denotes an unobserved target and is masked during fitting; infinity and genes with no observed targets are rejected; the training-gene-mean baseline uses observed values only",
            "nonfinite_scope": "all model inputs, predictions and selected external outcomes must be finite; unused publisher library metadata is audited separately",
            "duplicate_sample_or_gene_ids_allowed": False,
        },
        "replicate_policy": {
            "excluded_sample_library_pairs": [
                list(item) for item in EXCLUDED_SAMPLE_LIBRARIES
            ],
            "duplicate_exact_sample_library_gene_rows": "error",
            "remaining_libraries_for_same_sample_gene": "arithmetic mean",
            "distinct sample IDs including A/B suffixes": "kept distinct",
        },
        "endpoints": {
            "co_primary_models": list(CONDITIONS),
            "seed_aggregation": "arithmetic mean predictions across seeds 42, 43, 44 before outcome comparison",
            "primary": "per-organoid Spearman(model prediction, organoid LFC) minus Spearman(Broad 1103-cell training gene mean, organoid LFC), averaged across eligible organoids",
            "confidence_interval": f"tissue-stratified percentile bootstrap, {BOOTSTRAP_REPLICATES} replicates, seed {INFERENCE_SEED}",
            "hypothesis_test": f"two-sided paired sign-flip test; exact for n<=20, otherwise {SIGN_FLIP_MONTE_CARLO} Monte Carlo draws with add-one correction",
            "multiple_testing": "Holm correction across the two co-primary model tests",
            "support_rule": "Holm-adjusted p<0.05 and bootstrap 95% CI lower bound>0",
            "secondary": [
                "per-organoid top-5%-most-negative gene overlap and Jaccard versus outcome, with gene-mean baseline comparison",
                f"per-gene across-organoid Spearman for genes with at least {MIN_ORGANOIDS_PER_GENE} observations; BH within model",
                "seed-specific cellwise Spearman and tissue-stratified descriptive summaries",
            ],
            "top_k_count_rule": "ceil(0.05 * number of common genes), minimum 1",
            "top_k_tie_rule": "stable processed-gene order for equal scores",
            "not_primary": ["raw MSE", "raw MAE", "residual R2"],
            "scale_limitation": "Broad Chronos and organoid LFC are cross-platform scores with different scale construction; raw-scale error is not inferentially primary.",
        },
        "mutation_covariate_limitation": "External mutation inputs contain only publisher driver-mutation indicators, whereas Broad training uses a broader somatic mutation table.",
        "environment": _environment_inventory(),
    }
    body["lock_body_sha256"] = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    _commit_execution_bound_json(output, body, root=root)
    return body


def _load_freeze(path: Path) -> dict[str, Any]:
    payload_bytes, _ = _stable_read_bytes(path, label="protocol lock")
    return _load_freeze_bytes(payload_bytes)


def _load_freeze_bytes(payload_bytes: bytes) -> dict[str, Any]:
    payload = _parse_json_mapping(payload_bytes, "protocol lock")
    if payload.get("schema") != FREEZE_SCHEMA:
        raise ValidationError("unexpected protocol lock schema")
    digest = payload.get("lock_body_sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ValidationError("protocol lock body digest is invalid")
    unsigned = dict(payload)
    unsigned.pop("lock_body_sha256")
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != digest:
        raise ValidationError("protocol lock canonical body digest does not match")
    if (
        payload.get("write_once") is not True
        or payload.get("scientific_result") is not False
        or payload.get("selection_outcome_used") is not False
        or payload.get("numeric_outcome_argument_accepted_by_this_phase") is not False
        or payload.get("numeric_outcome_loaded_by_this_phase") is not False
        or payload.get("registration_status")
        != HISTORICAL_REVISION_PROVENANCE["status"]
        or payload.get("historical_revision_provenance")
        != HISTORICAL_REVISION_PROVENANCE
        or payload.get("required_formal_invocation") != _formal_invocation_constraint()
    ):
        raise ValidationError(
            "protocol lock has an invalid fixed-protocol revision state"
        )
    return payload


def _prevalidate_freeze_trust(
    path: Path,
    *,
    trusted_sha256: str,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the externally trusted lock using only the standard library."""
    trusted = trusted_sha256.strip().lower()
    if not SHA256_RE.fullmatch(trusted):
        raise ValidationError(
            "trusted freeze SHA-256 must be 64 lowercase hexadecimal characters"
        )
    payload, record = _stable_read_bytes(path, label="protocol lock", root=root)
    if record["sha256"] != trusted:
        raise ValidationError(
            "protocol lock does not match the independently supplied trusted freeze SHA-256"
        )
    return _load_freeze_bytes(payload), record


def _prevalidate_prediction_lock_trust(
    path: Path,
    *,
    trusted_sha256: str,
    freeze_record: Mapping[str, Any],
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind a prediction marker and its opaque artifacts before dependencies load."""
    trusted = trusted_sha256.strip().lower()
    if not SHA256_RE.fullmatch(trusted):
        raise ValidationError(
            "trusted prediction-lock SHA-256 must be 64 lowercase hexadecimal characters"
        )
    payload_bytes, record = _stable_read_bytes(path, label="prediction lock", root=root)
    if record["sha256"] != trusted:
        raise ValidationError(
            "prediction lock does not match the independently supplied trusted SHA-256"
        )
    payload = _parse_json_mapping(payload_bytes, "prediction lock")
    if (
        payload.get("schema") != PREDICTION_LOCK_SCHEMA
        or payload.get("numeric_outcome_argument_accepted_by_this_phase") is not False
        or payload.get("numeric_outcome_loaded_by_this_phase") is not False
        or payload.get("seed_aggregation_complete") is not True
    ):
        raise ValidationError("prediction lock has an invalid identity or phase state")
    if payload.get("freeze") != freeze_record:
        raise ValidationError(
            "prediction lock is not bound to the externally trusted protocol lock"
        )
    parent = path.resolve(strict=True).parent
    for key, filename, label in (
        ("predictions", "predictions.npz", "prediction archive"),
        ("training_records", "training-records.json", "training records"),
    ):
        artifact_record = payload.get(key)
        if not isinstance(artifact_record, Mapping):
            raise ValidationError(f"{label} record is missing")
        artifact_path = _verify_record(artifact_record, root=root, label=label)
        if artifact_path.parent != parent or artifact_path.name != filename:
            raise ValidationError(f"{label} escapes the prediction run directory")
    return payload, record


def validate_freeze(
    path: Path,
    *,
    root: Path = ROOT,
    trusted_sha256: str | None = None,
    expected_record: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if trusted_sha256 is not None:
        trusted_sha256 = trusted_sha256.strip().lower()
        if not SHA256_RE.fullmatch(trusted_sha256):
            raise ValidationError(
                "trusted freeze SHA-256 must be 64 lowercase hexadecimal characters"
            )
    _execution_source_records()
    freeze_bytes, freeze_record = _stable_read_bytes(
        path,
        label="protocol lock",
        root=root,
    )
    if trusted_sha256 is not None and freeze_record["sha256"] != trusted_sha256:
        raise ValidationError(
            "protocol lock does not match the independently supplied trusted freeze SHA-256"
        )
    if expected_record is not None:
        _assert_same_record(freeze_record, expected_record, label="protocol lock")
    freeze = _load_freeze_bytes(freeze_bytes)
    feature_audit = freeze.get("feature_audit")
    if not isinstance(feature_audit, dict):
        raise ValidationError("protocol lock lacks the non-outcome feature audit")
    if (
        feature_audit.get("numeric_outcome_loaded_by_this_phase") is not False
        or feature_audit.get("audit_passed") is not True
    ):
        raise ValidationError(
            "frozen non-outcome feature audit is not passing or phase-scoped"
        )
    if hashlib.sha256(_canonical_json_bytes(feature_audit)).hexdigest() != freeze.get(
        "feature_audit_canonical_sha256"
    ):
        raise ValidationError("frozen non-outcome feature audit digest does not match")
    canonical_inventories = {
        "training_inputs": list(TRAINING_INPUTS),
        "code_and_protocol": list(CODE_DEPENDENCIES),
    }
    for group_name, canonical_paths in canonical_inventories.items():
        records = freeze.get(group_name)
        if not isinstance(records, list) or len(records) != len(canonical_paths):
            raise ValidationError(
                f"protocol lock has a noncanonical {group_name} inventory"
            )
        observed_paths = [
            record.get("path") if isinstance(record, Mapping) else None
            for record in records
        ]
        if observed_paths != canonical_paths or len(observed_paths) != len(
            set(observed_paths)
        ):
            raise ValidationError(
                f"protocol lock has a noncanonical {group_name} path order or membership"
            )
        for index, record in enumerate(records):
            _verify_record(record, root=root, label=f"{group_name}[{index}]")
    bound_module_names = ("sanger_independent_validation", "embedding_provenance")
    bound_module_records = dict(
        zip(
            bound_module_names,
            _frozen_trusted_module_source_records(freeze, bound_module_names),
            strict=True,
        )
    )
    training_records = {
        record["path"]: record
        for record in freeze["training_inputs"]
        if isinstance(record, Mapping)
    }
    if (
        feature_audit.get("processed_gene_manifest")
        != training_records["data/processed/genes.txt"]
    ):
        raise ValidationError(
            "feature audit and training inventory disagree on the processed gene manifest"
        )
    if _embedding_bundle_provenance(
        root,
        expected_source_record=bound_module_records["embedding_provenance"],
    ) != freeze.get("embedding_bundle_provenance"):
        raise ValidationError(
            "ESM embedding bundle provenance differs from the protocol lock"
        )
    feature_files = freeze.get("feature_files")
    if not isinstance(feature_files, dict) or set(feature_files) != set(FEATURE_FILES):
        raise ValidationError("protocol lock feature inventory is incomplete")
    if feature_audit.get("feature_files") != feature_files:
        raise ValidationError(
            "feature audit and protocol lock disagree on external feature bindings"
        )
    for label, record in feature_files.items():
        _verify_record(record, root=root, label=f"feature_files.{label}")
    _, _, reconstructed_axis_bindings = _reconstruct_prediction_axes(freeze, root=root)
    if freeze.get("prediction_axis_bindings") != reconstructed_axis_bindings:
        raise ValidationError(
            "protocol lock prediction axes differ from frozen source order"
        )
    if feature_audit.get("prediction_axis_bindings") != reconstructed_axis_bindings:
        raise ValidationError(
            "feature audit prediction axes differ from frozen source order"
        )
    campaign_record = freeze.get("campaign")
    if not isinstance(campaign_record, Mapping):
        raise ValidationError("protocol lock lacks the canonical campaign record")
    campaign_path = _verify_record(campaign_record, root=root, label="campaign")
    report, consumed_campaign_record = _validate_campaign(
        campaign_path,
        root=root,
        expected_source_record=bound_module_records["sanger_independent_validation"],
    )
    _assert_same_record(
        consumed_campaign_record, campaign_record, label="canonical campaign"
    )
    if derive_training_plan(report, root=root) != freeze.get("training_plan"):
        raise ValidationError(
            "training plan no longer derives exactly from bound P1 folds"
        )
    figshare = freeze.get("figshare")
    if not isinstance(figshare, Mapping):
        raise ValidationError("protocol lock lacks Figshare metadata bindings")
    metadata_record = figshare.get("metadata", {})
    if not isinstance(metadata_record, Mapping):
        raise ValidationError(
            "protocol lock has an invalid Figshare metadata file record"
        )
    if metadata_record != feature_files["figshare_metadata"]:
        raise ValidationError(
            "Figshare metadata binding differs from the feature inventory"
        )
    metadata_path = _safe_project_path(metadata_record.get("path"), root)
    metadata_bytes, consumed_metadata_record = _stable_read_bytes(
        metadata_path,
        label="Figshare metadata",
        root=root,
    )
    _assert_same_record(
        consumed_metadata_record, metadata_record, label="Figshare metadata"
    )
    metadata = _load_figshare_metadata_bytes(metadata_bytes)
    observed_feature_metadata: dict[str, Any] = {}
    for label, expected in FEATURE_INPUTS.items():
        observed = _figshare_file(metadata, int(expected["id"]))
        for field in ("name", "bytes", "md5"):
            if observed[field] != expected[field]:
                raise ValidationError(
                    f"publisher feature metadata drift for {label}.{field}"
                )
        observed_feature_metadata[label] = observed
    if feature_audit.get("publisher_file_metadata") != observed_feature_metadata:
        raise ValidationError(
            "feature audit publisher metadata differs from pinned Figshare v1 metadata"
        )
    if _outcome_metadata(metadata) != figshare.get("publisher_outcome_files"):
        raise ValidationError("publisher outcome metadata no longer matches the lock")
    if feature_audit.get("publisher_outcome_file_metadata") != figshare.get(
        "publisher_outcome_files"
    ):
        raise ValidationError(
            "feature audit and protocol lock disagree on publisher outcome metadata"
        )
    if _official_source_record(root=root) != freeze.get("official_analysis_source"):
        raise ValidationError(
            "official Sanger analysis source differs from the protocol lock"
        )
    if _environment_inventory() != freeze.get("environment"):
        raise ValidationError(
            "execution environment differs from the frozen environment"
        )
    _execution_source_records()
    return freeze, freeze_record


def _load_external_features(
    features_dir: Path,
    base: Any,
    *,
    feature_records: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    for label in ("covariates", "rna", "driver_mutations"):
        payload, consumed_record = _stable_read_bytes(
            features_dir / FEATURE_FILES[label],
            label=f"external {label}",
            root=root,
        )
        _assert_same_record(
            consumed_record, feature_records[label], label=f"external {label}"
        )
        frames[label] = pd.read_csv(
            io.BytesIO(payload), dtype={"sample_ID": str}
        ).set_index("sample_ID")
    covariates = frames["covariates"]
    rna = frames["rna"]
    mutations = frames["driver_mutations"]
    external_organoid_ids = _require_unique(
        covariates.index, "external organoid sample_ID"
    )
    sample_ids = _require_unique(rna.index, "prediction sample_ID")
    broad_sample_ids = _require_unique(base.cell_ids, "Broad training sample_ID")
    exact_sample_id_overlap = sorted(
        set(broad_sample_ids).intersection(external_organoid_ids)
    )
    if exact_sample_id_overlap:
        raise ValidationError(
            "Broad training and external organoid sample IDs overlap exactly "
            f"({len(exact_sample_id_overlap)}): {exact_sample_id_overlap[:3]}"
        )
    if not set(sample_ids).issubset(covariates.index) or not set(sample_ids).issubset(
        mutations.index
    ):
        raise ValidationError(
            "prediction samples are absent from required external features"
        )
    covariates = covariates.reindex(sample_ids)
    rna = rna.reindex(sample_ids)
    mutations = mutations.reindex(sample_ids)

    rna_by_gene = {
        str(column)[:-4]: str(column)
        for column in rna.columns
        if str(column).endswith("_exp")
    }
    train_names = list(base.expression_gene_names)
    if len(train_names) != len(set(train_names)):
        raise ValidationError("Broad expression feature names are duplicated")
    train_mean = np.mean(np.asarray(base.expression_raw, dtype=np.float64), axis=0)
    if not np.isfinite(train_mean).all():
        raise ValidationError("Broad expression contains non-finite values")
    external_expression = np.broadcast_to(
        train_mean, (len(sample_ids), len(train_names))
    ).copy()
    matched_expression = 0
    for column_index, gene in enumerate(train_names):
        external_column = rna_by_gene.get(gene)
        if external_column is None:
            continue
        values = rna[external_column].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValidationError(f"external RNA contains non-finite values: {gene}")
        external_expression[:, column_index] = values
        matched_expression += 1
    if matched_expression < MIN_RNA_GENE_OVERLAP:
        raise ValidationError(
            "external RNA overlap with Broad expression is below the frozen gate"
        )

    mutation_columns = [str(column) for column in mutations.columns]
    mutation_values = mutations.to_numpy(dtype=np.float64, copy=False)
    if (
        not np.isfinite(mutation_values).all()
        or not np.isin(mutation_values, (0.0, 1.0)).all()
    ):
        raise ValidationError(
            "external driver mutation matrix is not finite binary data"
        )
    external_mutations = [
        {
            mutation_columns[index].removesuffix("_mut")
            for index in np.flatnonzero(mutation_values[row] == 1.0)
        }
        for row in range(len(sample_ids))
    ]
    tissues = covariates["tissue"].astype(str).tolist()
    try:
        lineages = np.asarray([LINEAGE_MAP[tissue] for tissue in tissues], dtype=str)
    except KeyError as exc:
        raise ValidationError(
            f"external tissue lacks a frozen lineage mapping: {exc}"
        ) from exc

    train_cells = len(broad_sample_ids)
    n_genes = len(base.genes)
    combined = type(base)(
        gene_effect=np.concatenate(
            [
                np.asarray(base.gene_effect, dtype=np.float32),
                np.full((n_genes, len(sample_ids)), np.nan, dtype=np.float32),
            ],
            axis=1,
        ),
        expression_raw=np.concatenate(
            [
                np.asarray(base.expression_raw, dtype=np.float32),
                external_expression.astype(np.float32),
            ],
            axis=0,
        ),
        mutations_by_cell=[*base.mutations_by_cell, *external_mutations],
        embeddings=np.asarray(base.embeddings, dtype=np.float32),
        edge_index=np.asarray(base.edge_index, dtype=np.int64),
        genes=list(base.genes),
        cell_ids=[*broad_sample_ids, *sample_ids],
        lineages=np.concatenate([np.asarray(base.lineages, dtype=str), lineages]),
        source_cell_indices=np.concatenate(
            [
                np.asarray(base.source_cell_indices, dtype=np.int64),
                np.full(len(sample_ids), -1, dtype=np.int64),
            ]
        ),
        source_gene_indices=np.asarray(base.source_gene_indices, dtype=np.int64),
        expression_gene_names=train_names,
    )
    external_indices = np.arange(
        train_cells, train_cells + len(sample_ids), dtype=np.int64
    )
    metadata = {
        "sample_ids": sample_ids,
        "sample_id_disjointness": {
            "comparison": "exact_identifier_equality_after_validation",
            "broad_training_sample_count": len(broad_sample_ids),
            "external_organoid_sample_count": len(external_organoid_ids),
            "external_prediction_sample_count": len(sample_ids),
            "exact_overlap_count": len(exact_sample_id_overlap),
            "exact_overlap_sample_ids": exact_sample_id_overlap,
            "disjoint": not exact_sample_id_overlap,
        },
        "tissues": tissues,
        "mapped_lineages": lineages.tolist(),
        "broad_expression_features": len(train_names),
        "matched_external_expression_features": matched_expression,
        "mean_imputed_external_expression_features": len(train_names)
        - matched_expression,
        "driver_mutation_features": len(mutation_columns),
        "mutation_covariate_limitation": "Only driver mutation indicators are available for organoids.",
    }
    return combined, external_indices, metadata


def _summarize_broad_training_outcomes(
    values: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the observed-value gene mean while auditing missing targets."""
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Broad training outcomes are not numeric") from exc
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValidationError("Broad training outcomes must be a non-empty matrix")
    if np.isinf(matrix).any():
        raise ValidationError("Broad training outcomes contain infinite values")
    finite = np.isfinite(matrix)
    observed_per_gene = finite.sum(axis=1, dtype=np.int64)
    if np.any(observed_per_gene == 0):
        raise ValidationError(
            "Broad training outcomes contain a gene with no observed values"
        )
    sums = np.where(finite, matrix, 0.0).sum(axis=1, dtype=np.float64)
    gene_mean = (sums / observed_per_gene).astype(np.float32)
    if not np.isfinite(gene_mean).all():
        raise ValidationError("Broad training gene means are not finite")
    audit = {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "finite_observations": int(finite.sum()),
        "missing_nan_targets": int(np.isnan(matrix).sum()),
        "minimum_observations_per_gene": int(observed_per_gene.min()),
        "maximum_observations_per_gene": int(observed_per_gene.max()),
        "missing_target_policy": "mask NaN targets during training and compute the baseline from observed values only",
        "infinite_target_policy": "reject",
    }
    return gene_mean, audit


def _predict_indices(
    core: Any,
    model: Any,
    config: Mapping[str, Any],
    fold_data: Any,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    torch = _load_trusted_runtime_module("torch")
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            mutation = torch.from_numpy(fold_data.mutation.matrix[selected]).to(device)
            expression = torch.from_numpy(fold_data.expression[selected]).to(device)
            lineage = torch.from_numpy(fold_data.lineage[selected]).to(device)
            tokens = torch.from_numpy(
                fold_data.mutation.tokens[selected, : core.K_TOKENS]
            ).to(device)
            token_mask = torch.from_numpy(
                fold_data.mutation.token_mask[selected, : core.K_TOKENS]
            ).to(device)
            prediction, _ = core._forward(
                model, config, mutation, expression, lineage, tokens, token_mask
            )
            values = prediction.float().cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(values).all():
                raise ValidationError(
                    "model produced NaN or infinite organoid predictions"
                )
            chunks.append(values)
    if not chunks:
        raise ValidationError("no external samples were available for prediction")
    return np.concatenate(chunks, axis=0)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError("refusing to serialize a non-finite value")
    return value


def _copy_bound_training_snapshot(
    records: Sequence[Mapping[str, Any]],
    *,
    snapshot_root: Path,
    root: Path,
) -> list[dict[str, Any]]:
    """Copy frozen training inputs into a private tree from verified open handles."""
    _create_bound_directory(snapshot_root, root=root)
    consumed: list[dict[str, Any]] = []
    for index, expected in enumerate(records):
        fields = _record_fields(expected, f"training_inputs[{index}]")
        source = _safe_project_path(fields["path"], root)
        destination = snapshot_root / Path(*PurePosixPath(fields["path"]).parts)
        digest = hashlib.sha256()
        size = 0

        def copy_and_validate(writer: Any) -> None:
            nonlocal size
            with source.open("rb") as reader:
                before = os.fstat(reader.fileno())
                if not stat_module.S_ISREG(before.st_mode):
                    raise ValidationError(
                        f"training input is not a regular file: {fields['path']}"
                    )
                while chunk := reader.read(4 * 1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    writer.write(chunk)
                after = os.fstat(reader.fileno())
                current = source.stat()
            if _stat_signature(before) != _stat_signature(after) or _stat_signature(
                after
            ) != _stat_signature(current):
                raise ValidationError(
                    f"training input changed during snapshot: {fields['path']}"
                )
            observed = {
                "path": fields["path"],
                "bytes": size,
                "sha256": digest.hexdigest(),
            }
            _assert_same_record(observed, fields, label=f"training_inputs[{index}]")

        try:
            _write_once_artifact(destination, copy_and_validate, root=root)
        except ValidationError:
            raise
        except OSError as exc:
            raise ValidationError(
                f"training input snapshot failed: {fields['path']}"
            ) from exc
        observed = {"path": fields["path"], "bytes": size, "sha256": digest.hexdigest()}
        snapshot_digest, snapshot_size = _stable_hash_and_size(destination, "sha256")
        if snapshot_digest != fields["sha256"] or snapshot_size != fields["bytes"]:
            raise ValidationError(
                f"private training snapshot is invalid: {fields['path']}"
            )
        consumed.append(observed)
    return consumed


def _restore_training_core_paths(
    core: Any, original_paths: tuple[Any, Any, Any, Any]
) -> None:
    core.ROOT, core.DATA, core.PROCESSED, core.DEPMAP = original_paths


def _remove_training_snapshot(snapshot_root: Path, run_dir: Path) -> None:
    if not snapshot_root.exists():
        return
    resolved_snapshot = snapshot_root.resolve(strict=True)
    if resolved_snapshot.parent != run_dir.resolve(
        strict=True
    ) or not resolved_snapshot.name.startswith(".training-input-snapshot-"):
        raise ValidationError("refusing to remove an unexpected training snapshot path")
    shutil.rmtree(resolved_snapshot)


def train_and_lock_predictions(
    *,
    freeze: Mapping[str, Any],
    freeze_record: Mapping[str, Any],
    run_dir: Path,
    features_dir: Path,
    device: str,
    root: Path = ROOT,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Train all six full-cohort models in the outcome-free prediction phase."""
    _require_lock_predictions_single_flight()
    expected_gene_ids, expected_sample_ids, axis_bindings = (
        _reconstruct_prediction_axes(freeze, root=root)
    )
    executed_module_sources = _assert_trusted_modules_match_freeze(
        freeze, TRAINING_LOCAL_MODULES, root=root
    )
    protocol = _TRUSTED_LOCAL_MODULES["corrected_protocol"][0]
    core = _TRUSTED_LOCAL_MODULES["run_corrected_experiments"][0]
    core_binding = _TRUSTED_LOCAL_MODULES.get("run_corrected_experiments")
    if core_binding is None or core_binding[0] is not core:
        raise ValidationError("training core lost trusted execution provenance")
    core_baseline_fingerprint = core_binding[2]
    torch = _load_trusted_runtime_module("torch")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValidationError("CUDA was requested but is unavailable")
    snapshot_root = run_dir / f".training-input-snapshot-{uuid.uuid4().hex}"
    consumed_training_inputs: list[dict[str, Any]] = []
    original_paths = (core.ROOT, core.DATA, core.PROCESSED, core.DEPMAP)
    base: Any = None
    temporary_fingerprint: str | None = None
    core_cleanup_required = False
    primary_failure: BaseException | None = None
    cleanup_failures: list[CleanupFailure] = []
    try:
        consumed_training_inputs = _copy_bound_training_snapshot(
            freeze["training_inputs"],
            snapshot_root=snapshot_root,
            root=root,
        )
        core_cleanup_required = True
        core.ROOT = snapshot_root
        core.DATA = snapshot_root / "data"
        core.PROCESSED = core.DATA / "processed"
        core.DEPMAP = core.DATA / "depmap"
        temporary_fingerprint = _trusted_module_namespace_fingerprint(
            core, name="run_corrected_experiments"
        )
        _assert_trusted_module_namespace(
            core,
            name="run_corrected_experiments",
            expected_fingerprint=temporary_fingerprint,
        )
        base = core.load_base_data(
            pilot=False, seed=42, pilot_cells=96, pilot_genes=128
        )
    except BaseException as exc:
        primary_failure = exc

    if core_cleanup_required:
        if temporary_fingerprint is not None:
            _capture_cleanup_failure(
                cleanup_failures,
                "temporary training-core namespace validation",
                lambda: _assert_trusted_module_namespace(
                    core,
                    name="run_corrected_experiments",
                    expected_fingerprint=temporary_fingerprint,
                ),
            )
        _capture_cleanup_failure(
            cleanup_failures,
            "initial training-core path restoration",
            lambda: _restore_training_core_paths(core, original_paths),
        )
        _capture_cleanup_failure(
            cleanup_failures,
            "second-chance training-core path restoration",
            lambda: _restore_training_core_paths(core, original_paths),
        )
        _capture_cleanup_failure(
            cleanup_failures,
            "training-core baseline namespace validation",
            lambda: _assert_trusted_module_namespace(
                core,
                name="run_corrected_experiments",
                expected_fingerprint=core_baseline_fingerprint,
            ),
        )
    _capture_cleanup_failure(
        cleanup_failures,
        "private training-snapshot removal",
        lambda: _remove_training_snapshot(snapshot_root, run_dir),
    )
    if primary_failure is not None or cleanup_failures:
        _raise_after_cleanup(primary_failure, cleanup_failures)
    if len(base.cell_ids) != 1103:
        raise ValidationError(
            f"full Broad cohort must contain 1103 cells, observed {len(base.cell_ids)}"
        )
    if list(base.genes) != expected_gene_ids:
        raise ValidationError(
            "loaded Broad gene order differs from frozen genes.txt line order"
        )
    training_gene_mean, training_outcome_audit = _summarize_broad_training_outcomes(
        base.gene_effect
    )
    combined, external_indices, feature_metadata = _load_external_features(
        features_dir,
        base,
        feature_records=freeze["feature_files"],
        root=root,
    )
    if feature_metadata["sample_ids"] != expected_sample_ids:
        raise ValidationError(
            "external prediction cohort differs from the complete frozen RNA sample order"
        )
    training_indices = np.arange(len(base.cell_ids), dtype=np.int64)
    plan_rows = freeze["training_plan"]["plans"]
    by_key = {(row["condition"], int(row["training_seed"])): row for row in plan_rows}
    arrays: dict[str, np.ndarray] = {
        "sample_ids": np.asarray(feature_metadata["sample_ids"], dtype=str),
        "gene_ids": np.asarray(base.genes, dtype=str),
        "training_gene_mean": training_gene_mean,
    }
    training_records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        seed_predictions: list[np.ndarray] = []
        for seed in SEEDS:
            _assert_trusted_modules_match_freeze(
                freeze, TRAINING_LOCAL_MODULES, root=root
            )
            plan = by_key.get((condition, seed))
            if plan is None:
                raise ValidationError(
                    f"frozen training plan is incomplete: {condition}, {seed}"
                )
            refit_seed = protocol.derive_seed(seed, "organoid-full-refit", condition)
            fold_data = core.fit_fold_data(
                combined,
                training_indices,
                seed=protocol.derive_seed(refit_seed, "outer-preprocess"),
                n_expr_pc=int(freeze["training_plan"]["n_expr_pc"]),
                top_mut=int(freeze["training_plan"]["top_mut"]),
            )
            protocol.set_global_seed(refit_seed)
            config = dict(plan["selected_config"])
            model = core.build_model(config, combined, fold_data)
            runtime_before_fit = _training_runtime_record(torch, device)
            model, training = core._train_epochs(
                model,
                config,
                combined,
                fold_data,
                training_indices,
                epochs=int(plan["selected_epoch"]),
                batch_size=int(freeze["training_plan"]["batch_size"]),
                gene_subsample=int(freeze["training_plan"]["gene_subsample"]),
                device=device,
                seed=refit_seed,
                validation_idx=None,
                patience=5,
                scheduler_horizon=int(freeze["training_plan"]["scheduler_horizon"]),
            )
            prediction = _predict_indices(
                core,
                model,
                config,
                fold_data,
                external_indices,
                batch_size=int(freeze["training_plan"]["batch_size"]),
                device=device,
            )
            runtime_after_prediction = _training_runtime_record(torch, device)
            if runtime_before_fit != runtime_after_prediction:
                raise ValidationError(
                    "training runtime changed during a full refit/prediction pair"
                )
            slug = condition.lower().replace("-", "_")
            arrays[f"{slug}_seed_{seed}"] = prediction
            seed_predictions.append(prediction)
            training_records.append(
                {
                    "condition": condition,
                    "training_seed": seed,
                    "full_refit_seed": int(refit_seed),
                    "config": config,
                    "epochs": int(plan["selected_epoch"]),
                    "runtime_before_fit": runtime_before_fit,
                    "runtime_after_prediction": runtime_after_prediction,
                    "training": _jsonable(training),
                    "preprocessing": _jsonable(
                        core.preprocessing_manifest(fold_data, external_indices)
                    ),
                }
            )
            del model, fold_data
            if device == "cuda":
                torch.cuda.empty_cache()
        arrays[f"{condition.lower().replace('-', '_')}_aggregate"] = np.mean(
            np.stack(seed_predictions, axis=0), axis=0, dtype=np.float64
        ).astype(np.float32)
    _assert_trusted_modules_match_freeze(freeze, TRAINING_LOCAL_MODULES, root=root)
    expected_shape = (len(feature_metadata["sample_ids"]), len(base.genes))
    for name, array in arrays.items():
        if name in {"sample_ids", "gene_ids"}:
            continue
        if name == "training_gene_mean":
            if array.shape != (len(base.genes),):
                raise ValidationError("training gene mean shape is invalid")
        elif array.shape != expected_shape:
            raise ValidationError(f"prediction array has the wrong shape: {name}")
        if not np.isfinite(array).all():
            raise ValidationError(
                f"prediction artifact contains non-finite values: {name}"
            )

    prediction_path = run_dir / "predictions.npz"
    _write_once_artifact(
        prediction_path,
        lambda handle: np.savez_compressed(handle, **arrays),
        root=root,
    )
    training_payload = {
        "schema": TRAINING_RECORDS_SCHEMA,
        "numeric_outcome_argument_accepted_by_this_phase": False,
        "numeric_outcome_loaded_by_this_phase": False,
        "broad_training_cells": len(base.cell_ids),
        "broad_training_outcomes": training_outcome_audit,
        "external_prediction_samples": len(feature_metadata["sample_ids"]),
        "external_feature_processing": feature_metadata,
        "prediction_axis_bindings": axis_bindings,
        "consumed_training_inputs": consumed_training_inputs,
        "executed_local_module_sources": executed_module_sources,
        "private_training_snapshot_removed_after_load": True,
        "device": device,
        "runtime": _training_runtime_record(torch, device),
        "models": training_records,
    }
    _atomic_json(run_dir / "training-records.json", training_payload, root=root)
    prediction_record = _record_file(prediction_path, root)
    training_record = _record_file(run_dir / "training-records.json", root)
    lock = {
        "schema": PREDICTION_LOCK_SCHEMA,
        "created_utc": _utc_now(),
        "numeric_outcome_argument_accepted_by_this_phase": False,
        "numeric_outcome_loaded_by_this_phase": False,
        "freeze": dict(freeze_record),
        "predictions": prediction_record,
        "training_records": training_record,
        "arrays": {
            name: {"dtype": str(array.dtype), "shape": list(array.shape)}
            for name, array in arrays.items()
        },
        "prediction_axis_bindings": axis_bindings,
        "seed_aggregation_complete": True,
    }
    return arrays, training_payload, lock


def _normalise_library(value: Any) -> str:
    text = str(value).strip()
    aliases = {"v1.1": "v1_1", "v1_1": "v1_1", "minlib": "minLib", "minLib": "minLib"}
    return aliases.get(text, text)


def normalise_outcome_long(
    frame: pd.DataFrame,
    *,
    allowed_samples: Sequence[str],
) -> OutcomeMatrix:
    """Validate, apply fixed replicate exclusions, and aggregate a long outcome table."""
    _require_unique(frame.columns, "long outcome column")
    required = {"sample_ID", "gene", "library", "LFC"}
    if not required.issubset(frame.columns):
        raise ValidationError(
            f"long outcome table lacks columns: {sorted(required - set(frame.columns))}"
        )
    selected = frame[["sample_ID", "gene", "library", "LFC"]].copy()
    selected["sample_ID"] = _clean_identifiers(selected["sample_ID"], "outcome sample")
    selected["gene"] = _clean_identifiers(selected["gene"], "outcome gene")
    selected["library"] = [
        _normalise_library(value)
        for value in _clean_identifiers(selected["library"], "outcome library")
    ]
    try:
        selected["LFC"] = pd.to_numeric(selected["LFC"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValidationError("outcome LFC values must be numeric") from exc
    if not np.isfinite(selected["LFC"].to_numpy(dtype=np.float64)).all():
        raise ValidationError("outcome contains NaN or infinite values")
    selected = selected[selected["sample_ID"].isin(set(allowed_samples))].copy()
    if selected.empty:
        raise ValidationError("outcome contains no feature-eligible organoids")
    if selected.duplicated(["sample_ID", "library", "gene"]).any():
        raise ValidationError("outcome contains duplicate sample/library/gene rows")
    exclusions = set(EXCLUDED_SAMPLE_LIBRARIES)
    exclude_mask = np.fromiter(
        (
            (sample, library) in exclusions
            for sample, library in zip(selected["sample_ID"], selected["library"])
        ),
        dtype=bool,
        count=len(selected),
    )
    excluded_rows = int(exclude_mask.sum())
    selected = selected.loc[~exclude_mask]
    library_counts = selected.groupby(["sample_ID", "gene"], sort=False)[
        "library"
    ].nunique()
    libraries_aggregated = int((library_counts > 1).sum())
    collapsed = selected.groupby(["sample_ID", "gene"], as_index=False, sort=True)[
        "LFC"
    ].mean()
    if not np.isfinite(collapsed["LFC"].to_numpy(dtype=np.float64)).all():
        raise ValidationError("aggregated outcome contains non-finite values")
    sample_ids = sorted(collapsed["sample_ID"].unique().tolist())
    gene_ids = sorted(collapsed["gene"].unique().tolist())
    if len(sample_ids) != len(set(sample_ids)) or len(gene_ids) != len(set(gene_ids)):
        raise ValidationError("outcome pivot identities are duplicated")
    pivot = collapsed.pivot(index="sample_ID", columns="gene", values="LFC").reindex(
        index=sample_ids, columns=gene_ids
    )
    return OutcomeMatrix(
        sample_ids=tuple(sample_ids),
        gene_ids=tuple(gene_ids),
        values=pivot.to_numpy(dtype=np.float64),
        libraries_aggregated=libraries_aggregated,
        excluded_rows=excluded_rows,
    )


def _wide_to_long(frame: pd.DataFrame, allowed_samples: Sequence[str]) -> pd.DataFrame:
    _require_unique(frame.columns, "wide outcome column")
    gene_column = next(
        (
            name
            for name in ("gene", "symbol", "gene_name", "Gene")
            if name in frame.columns
        ),
        None,
    )
    working = frame.copy()
    if gene_column is None:
        if isinstance(working.index, pd.RangeIndex):
            raise ValidationError("wide outcome has no gene column or gene row names")
        working.insert(0, "gene", working.index.astype(str))
        gene_column = "gene"
    genes = _require_unique(working[gene_column], "outcome gene")
    allowed = sorted({str(value) for value in allowed_samples}, key=len, reverse=True)
    mapped: list[tuple[str, str, str]] = []
    for column in working.columns:
        if column == gene_column:
            continue
        name = str(column)
        match = next(
            (
                sample
                for sample in allowed
                if name.startswith((sample + ".", sample + "_"))
            ),
            None,
        )
        if match is None:
            continue
        separator = name[len(match)]
        remainder = name[len(match) + 1 :]
        if separator not in {".", "_"} or not remainder:
            raise ValidationError(
                f"wide outcome column lacks a sample/library separator: {name}"
            )
        library = _normalise_library(remainder)
        if library not in {"minLib", "v1_1"}:
            raise ValidationError(
                f"wide outcome column lacks an explicit canonical library: {name}"
            )
        mapped.append((name, match, library))
    if not mapped:
        raise ValidationError(
            "wide outcome contains no recognized sample/library columns"
        )
    rows: list[pd.DataFrame] = []
    for column, sample, library in mapped:
        rows.append(
            pd.DataFrame(
                {
                    "sample_ID": sample,
                    "gene": genes,
                    "library": library,
                    "LFC": working[column].to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _load_outcome(source: Any, *, allowed_samples: Sequence[str]) -> OutcomeMatrix:
    """Parse an RData stream after callers establish and validate the prediction lock."""
    loaded_pyreadr = _load_trusted_runtime_module("pyreadr")
    try:
        objects = loaded_pyreadr.read_r(source)
    except Exception as exc:
        raise ValidationError(f"cannot read the RData outcome: {exc}") from exc
    candidates: list[OutcomeMatrix] = []
    errors: list[str] = []
    for name, frame in objects.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        try:
            if {"sample_ID", "gene", "library", "LFC"}.issubset(frame.columns):
                long = frame
            else:
                long = _wide_to_long(frame, allowed_samples)
            candidates.append(
                normalise_outcome_long(long, allowed_samples=allowed_samples)
            )
        except ValidationError as exc:
            errors.append(f"{name}: {exc}")
    if len(candidates) != 1:
        detail = "; ".join(errors[:3])
        raise ValidationError(
            f"expected exactly one usable outcome object, observed {len(candidates)}; {detail}"
        )
    return candidates[0]


def _load_bound_outcome(
    path: Path,
    *,
    expected_record: Mapping[str, Any],
    publisher_metadata: Mapping[str, Any],
    allowed_samples: Sequence[str],
    root: Path,
) -> OutcomeMatrix:
    """Hash, parse and re-hash one stable open outcome file instance."""
    record = _record_fields(expected_record, "outcome snapshot")
    if (
        not isinstance(record["path"], str)
        or not isinstance(record["bytes"], int)
        or isinstance(record["bytes"], bool)
        or record["bytes"] < 0
        or not isinstance(record["sha256"], str)
        or not SHA256_RE.fullmatch(record["sha256"])
    ):
        raise ValidationError("outcome snapshot file record is invalid")
    expected_path = _safe_project_path(record["path"], root)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValidationError("outcome snapshot is unavailable") from exc
    if resolved != expected_path or not resolved.is_file():
        raise ValidationError("outcome snapshot path differs from its binding")

    publisher_bytes = publisher_metadata.get("bytes")
    publisher_md5 = publisher_metadata.get("md5")
    if (
        not isinstance(publisher_bytes, int)
        or isinstance(publisher_bytes, bool)
        or publisher_bytes < 0
        or not isinstance(publisher_md5, str)
        or not MD5_RE.fullmatch(publisher_md5)
    ):
        raise ValidationError("frozen publisher outcome metadata is invalid")

    try:
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(before.st_mode):
                raise ValidationError("outcome snapshot is not a regular file")
            sha256_before, md5_before, size_before = _hash_open_handle(handle)
            if (
                size_before != record["bytes"]
                or sha256_before != record["sha256"]
                or size_before != publisher_bytes
                or md5_before != publisher_md5
            ):
                raise ValidationError(
                    "outcome snapshot differs from its frozen bindings"
                )
            outcome = _load_outcome(handle, allowed_samples=allowed_samples)
            if handle.closed:
                raise ValidationError(
                    "RData parser closed the protected outcome file instance"
                )
            sha256_after, md5_after, size_after = _hash_open_handle(handle)
            after = os.fstat(handle.fileno())
        current = resolved.stat()
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(
            "outcome snapshot cannot be read as a stable file instance"
        ) from exc
    if (
        _stat_signature(before) != _stat_signature(after)
        or _stat_signature(after) != _stat_signature(current)
        or (sha256_after, md5_after, size_after)
        != (sha256_before, md5_before, size_before)
    ):
        raise ValidationError(
            "outcome snapshot changed while its protected instance was parsed"
        )
    return outcome


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1 or len(left) < 3:
        raise ValidationError(
            "Spearman inputs must be paired vectors of length at least three"
        )
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValidationError("Spearman inputs must be finite")
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        raise ValidationError("Spearman correlation is undefined for a constant vector")
    value = float(stats.spearmanr(left, right).statistic)
    if not math.isfinite(value):
        raise ValidationError("Spearman correlation is non-finite")
    return value


def _top_overlap(
    prediction: np.ndarray, outcome: np.ndarray, fraction: float = TOP_FRACTION
) -> tuple[int, float, int]:
    if prediction.shape != outcome.shape or prediction.ndim != 1:
        raise ValidationError("top-overlap vectors must share one-dimensional shape")
    top_n = max(1, math.ceil(len(prediction) * fraction))
    predicted = set(np.argsort(prediction, kind="stable")[:top_n].tolist())
    observed = set(np.argsort(outcome, kind="stable")[:top_n].tolist())
    overlap = len(predicted & observed)
    union = len(predicted | observed)
    return overlap, float(overlap / union), top_n


def _stratified_bootstrap(
    values: np.ndarray, tissues: Sequence[str]
) -> tuple[float, float]:
    if len(values) != len(tissues) or not np.isfinite(values).all():
        raise ValidationError("bootstrap inputs are invalid")
    tissue_array = np.asarray(tissues, dtype=str)
    observed_tissues = set(tissue_array.tolist())
    required_tissues = set(LINEAGE_MAP)
    if observed_tissues != required_tissues:
        missing = sorted(required_tissues - observed_tissues)
        unexpected = sorted(observed_tissues - required_tissues)
        raise ValidationError(
            "bootstrap tissues must contain all and only the fixed strata; "
            f"missing={missing}, unexpected={unexpected}"
        )
    rng = np.random.default_rng(INFERENCE_SEED)
    groups = [np.flatnonzero(tissue_array == tissue) for tissue in LINEAGE_MAP]
    if any(len(group) == 0 for group in groups):
        raise ValidationError("bootstrap contains an empty tissue stratum")
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        estimates[replicate] = float(np.mean(values[sampled]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _sign_flip_pvalue(values: np.ndarray) -> tuple[float, str, int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValidationError("sign-flip inputs must be a non-empty finite vector")
    observed = abs(float(np.mean(values)))
    tolerance = np.finfo(np.float64).eps * max(1.0, observed) * 8
    if len(values) <= 20:
        total = 1 << len(values)
        exceed = 0
        for start in range(0, total, 65_536):
            codes = np.arange(start, min(start + 65_536, total), dtype=np.uint64)
            bits = (
                (codes[:, None] >> np.arange(len(values), dtype=np.uint64)) & 1
            ).astype(np.float64)
            signs = bits * 2.0 - 1.0
            estimates = np.abs((signs @ values) / len(values))
            exceed += int(np.sum(estimates + tolerance >= observed))
        return float(exceed / total), "exact", total
    rng = np.random.default_rng(INFERENCE_SEED + 1)
    exceed = 0
    completed = 0
    while completed < SIGN_FLIP_MONTE_CARLO:
        size = min(10_000, SIGN_FLIP_MONTE_CARLO - completed)
        signs = (
            rng.integers(0, 2, size=(size, len(values)), dtype=np.int8).astype(
                np.float64
            )
            * 2.0
            - 1.0
        )
        estimates = np.abs((signs @ values) / len(values))
        exceed += int(np.sum(estimates + tolerance >= observed))
        completed += size
    return (
        float((exceed + 1) / (SIGN_FLIP_MONTE_CARLO + 1)),
        "monte_carlo_add_one",
        SIGN_FLIP_MONTE_CARLO,
    )


def _holm(pvalues: Sequence[float]) -> list[float]:
    indexed = sorted(
        enumerate(float(value) for value in pvalues), key=lambda item: item[1]
    )
    adjusted = [1.0] * len(indexed)
    running = 0.0
    total = len(indexed)
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted


def _bh(pvalues: Sequence[float]) -> list[float]:
    indexed = sorted(
        enumerate(float(value) for value in pvalues), key=lambda item: item[1]
    )
    total = len(indexed)
    adjusted = [1.0] * total
    running = 1.0
    for reverse in range(total - 1, -1, -1):
        index, value = indexed[reverse]
        running = min(running, value * total / (reverse + 1))
        adjusted[index] = min(1.0, running)
    return adjusted


def _align_analysis_data(
    predictions: Mapping[str, np.ndarray],
    outcome: OutcomeMatrix,
    tissue_by_sample: Mapping[str, str],
) -> tuple[
    list[str], list[str], list[str], np.ndarray, np.ndarray, dict[str, np.ndarray]
]:
    prediction_samples = [str(value) for value in predictions["sample_ids"].tolist()]
    prediction_genes = [str(value) for value in predictions["gene_ids"].tolist()]
    _require_unique(prediction_samples, "prediction sample_ID")
    _require_unique(prediction_genes, "prediction gene")
    outcome_sample_index = {
        value: index for index, value in enumerate(outcome.sample_ids)
    }
    outcome_gene_index = {value: index for index, value in enumerate(outcome.gene_ids)}
    eligible_samples = [
        sample for sample in prediction_samples if sample in outcome_sample_index
    ]
    common_genes = [gene for gene in prediction_genes if gene in outcome_gene_index]
    if not eligible_samples or not common_genes:
        raise ValidationError("prediction and outcome identities do not overlap")
    outcome_values = outcome.values[
        np.ix_(
            np.asarray(
                [outcome_sample_index[sample] for sample in eligible_samples],
                dtype=np.int64,
            ),
            np.asarray(
                [outcome_gene_index[gene] for gene in common_genes], dtype=np.int64
            ),
        )
    ]
    complete = np.isfinite(outcome_values).all(axis=0)
    common_genes = [gene for gene, keep in zip(common_genes, complete) if bool(keep)]
    outcome_values = outcome_values[:, complete]
    if len(eligible_samples) < MIN_PRIMARY_ORGANOIDS:
        raise ValidationError(
            f"only {len(eligible_samples)} eligible organoids; require {MIN_PRIMARY_ORGANOIDS}"
        )
    if len(common_genes) < MIN_PRIMARY_GENES:
        raise ValidationError(
            f"only {len(common_genes)} complete common genes; require {MIN_PRIMARY_GENES}"
        )
    if outcome_values.shape[1] < MIN_GENES_PER_ORGANOID:
        raise ValidationError(
            "eligible organoids do not meet the per-organoid gene gate"
        )
    if not np.isfinite(outcome_values).all():
        raise ValidationError("selected analysis outcome contains non-finite values")
    sample_index = {value: index for index, value in enumerate(prediction_samples)}
    gene_index = {value: index for index, value in enumerate(prediction_genes)}
    rows = np.asarray(
        [sample_index[value] for value in eligible_samples], dtype=np.int64
    )
    columns = np.asarray([gene_index[value] for value in common_genes], dtype=np.int64)
    aligned: dict[str, np.ndarray] = {}
    for name, array in predictions.items():
        if name in {"sample_ids", "gene_ids"}:
            continue
        values = np.asarray(array)
        if name == "training_gene_mean":
            aligned[name] = values[columns].astype(np.float64)
        else:
            aligned[name] = values[np.ix_(rows, columns)].astype(np.float64)
        if not np.isfinite(aligned[name]).all():
            raise ValidationError(
                f"selected prediction data contain non-finite values: {name}"
            )
    tissues = [str(tissue_by_sample[sample]) for sample in eligible_samples]
    observed_tissues = set(tissues)
    required_tissues = set(LINEAGE_MAP)
    if observed_tissues != required_tissues:
        missing = sorted(required_tissues - observed_tissues)
        unexpected = sorted(observed_tissues - required_tissues)
        raise ValidationError(
            "eligible samples must contain all and only the fixed tissue strata; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return eligible_samples, common_genes, tissues, outcome_values, rows, aligned


def compute_analysis(
    predictions: Mapping[str, np.ndarray],
    outcome: OutcomeMatrix,
    *,
    tissue_by_sample: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    samples, genes, tissues, outcome_values, _, aligned = _align_analysis_data(
        predictions, outcome, tissue_by_sample
    )
    baseline = aligned["training_gene_mean"]
    cell_rows: list[dict[str, Any]] = []
    primary: list[dict[str, Any]] = []
    primary_deltas: dict[str, np.ndarray] = {}
    baseline_rhos = np.asarray(
        [_spearman(baseline, row) for row in outcome_values], dtype=np.float64
    )
    for condition in CONDITIONS:
        slug = condition.lower().replace("-", "_")
        matrices = [("aggregate", aligned[f"{slug}_aggregate"])]
        matrices.extend((str(seed), aligned[f"{slug}_seed_{seed}"]) for seed in SEEDS)
        for seed_label, matrix in matrices:
            for row_index, sample in enumerate(samples):
                rho_model = _spearman(matrix[row_index], outcome_values[row_index])
                model_overlap, model_jaccard, top_n = _top_overlap(
                    matrix[row_index], outcome_values[row_index]
                )
                baseline_overlap, baseline_jaccard, _ = _top_overlap(
                    baseline, outcome_values[row_index]
                )
                cell_rows.append(
                    {
                        "sample_ID": sample,
                        "tissue": tissues[row_index],
                        "condition": condition,
                        "seed": seed_label,
                        "n_genes": len(genes),
                        "spearman_model": rho_model,
                        "spearman_gene_mean_baseline": float(baseline_rhos[row_index]),
                        "spearman_delta": rho_model - float(baseline_rhos[row_index]),
                        "top_n": top_n,
                        "top_overlap_model": model_overlap,
                        "top_jaccard_model": model_jaccard,
                        "top_overlap_baseline": baseline_overlap,
                        "top_jaccard_baseline": baseline_jaccard,
                        "top_jaccard_delta": model_jaccard - baseline_jaccard,
                    }
                )
        aggregate_rows = [
            row
            for row in cell_rows
            if row["condition"] == condition and row["seed"] == "aggregate"
        ]
        deltas = np.asarray(
            [row["spearman_delta"] for row in aggregate_rows], dtype=np.float64
        )
        primary_deltas[condition] = deltas
        ci_low, ci_high = _stratified_bootstrap(deltas, tissues)
        pvalue, test_mode, draws = _sign_flip_pvalue(deltas)
        tissue_summary = []
        for tissue in LINEAGE_MAP:
            indices = np.flatnonzero(np.asarray(tissues, dtype=str) == tissue)
            tissue_summary.append(
                {
                    "tissue": tissue,
                    "n_organoids": len(indices),
                    "mean_spearman_delta": float(np.mean(deltas[indices])),
                    "median_spearman_delta": float(np.median(deltas[indices])),
                }
            )
        primary.append(
            {
                "condition": condition,
                "n_organoids": len(deltas),
                "n_common_complete_genes": len(genes),
                "mean_spearman_model": float(
                    np.mean([row["spearman_model"] for row in aggregate_rows])
                ),
                "mean_spearman_gene_mean_baseline": float(np.mean(baseline_rhos)),
                "mean_paired_spearman_delta": float(np.mean(deltas)),
                "median_paired_spearman_delta": float(np.median(deltas)),
                "bootstrap_95_ci": [ci_low, ci_high],
                "sign_flip_p_two_sided": pvalue,
                "sign_flip_mode": test_mode,
                "sign_flip_draws": draws,
                "tissue_descriptive": tissue_summary,
            }
        )
    adjusted = _holm([row["sign_flip_p_two_sided"] for row in primary])
    for row, value in zip(primary, adjusted):
        row["holm_adjusted_p"] = value
        row["primary_supported"] = bool(
            value < 0.05 and row["bootstrap_95_ci"][0] > 0.0
        )

    gene_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        slug = condition.lower().replace("-", "_")
        matrix = aligned[f"{slug}_aggregate"]
        condition_rows: list[dict[str, Any]] = []
        for gene_index, gene in enumerate(genes):
            predicted = matrix[:, gene_index]
            observed = outcome_values[:, gene_index]
            if (
                len(observed) < MIN_ORGANOIDS_PER_GENE
                or np.ptp(predicted) == 0
                or np.ptp(observed) == 0
            ):
                continue
            result = stats.spearmanr(predicted, observed)
            rho = float(result.statistic)
            pvalue = float(result.pvalue)
            if not math.isfinite(rho) or not math.isfinite(pvalue):
                continue
            condition_rows.append(
                {
                    "condition": condition,
                    "gene": gene,
                    "n_organoids": len(observed),
                    "spearman_rho": rho,
                    "p_two_sided": pvalue,
                }
            )
        qvalues = _bh([row["p_two_sided"] for row in condition_rows])
        for row, qvalue in zip(condition_rows, qvalues):
            row["q_bh_within_model"] = qvalue
        gene_rows.extend(condition_rows)

    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "scientific_result": True,
        "validation_scope": "independent patient-derived organoid cohort and experimental platform",
        "phase_binding": "analysis consumed a prediction lock whose SHA-256 was supplied externally to the analysis command; this integrity binding does not independently establish real-world chronology",
        "cohort": {
            "eligible_organoids": len(samples),
            "complete_common_genes": len(genes),
            "tissue_counts": {
                key: int(value) for key, value in sorted(Counter(tissues).items())
            },
            "excluded_outcome_rows": outcome.excluded_rows,
            "sample_gene_groups_aggregating_multiple_libraries": outcome.libraries_aggregated,
        },
        "primary_endpoint": {
            "definition": "mean paired per-organoid Spearman improvement over the Broad 1103-cell training gene-mean baseline",
            "seed_aggregation": "arithmetic mean across seeds 42, 43, 44 before outcome comparison",
            "models": primary,
            "familywise_correction": "Holm across DepPLM and C-PLM-MLP",
        },
        "secondary_endpoints": {
            "top_fraction": TOP_FRACTION,
            "top_count_rule": "ceil(top_fraction * number of common genes), minimum 1",
            "top_tie_rule": "stable processed-gene order for equal scores",
            "per_gene_minimum_organoids": MIN_ORGANOIDS_PER_GENE,
            "per_gene_multiple_testing": "Benjamini-Hochberg separately within each model",
            "cellwise_rows": len(cell_rows),
            "gene_rows": len(gene_rows),
        },
        "scale_guardrail": "Raw MSE, raw MAE and residual R2 are not computed because Chronos and organoid LFC have different platform-specific scales.",
        "mutation_covariate_limitation": "Organoid predictions use only the available driver-mutation indicators; Broad training uses broader somatic mutation calls.",
        "cellwise_rows_canonical_sha256": hashlib.sha256(
            _canonical_json_bytes(cell_rows)
        ).hexdigest(),
        "gene_rows_canonical_sha256": hashlib.sha256(
            _canonical_json_bytes(gene_rows)
        ).hexdigest(),
        "conclusion": {
            row["condition"]: (
                "prespecified primary support criterion met"
                if row["primary_supported"]
                else "prespecified primary support criterion not met"
            )
            for row in primary
        },
    }
    return analysis, cell_rows, gene_rows


CELLWISE_FIELDS = (
    "sample_ID",
    "tissue",
    "condition",
    "seed",
    "n_genes",
    "spearman_model",
    "spearman_gene_mean_baseline",
    "spearman_delta",
    "top_n",
    "top_overlap_model",
    "top_jaccard_model",
    "top_overlap_baseline",
    "top_jaccard_baseline",
    "top_jaccard_delta",
)
GENE_FIELDS = (
    "condition",
    "gene",
    "n_organoids",
    "spearman_rho",
    "p_two_sided",
    "q_bh_within_model",
)


def _path_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=True))
    except ValueError:
        return False
    return True


def _new_run_directory(output_root: Path, run_id: str | None, *, root: Path) -> Path:
    if run_id is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        run_id = f"{timestamp}-organoid-{uuid.uuid4().hex[:12]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", run_id):
        raise ValidationError("run ID contains unsafe characters or is too long")
    _, bound_root_path, _ = _lexical_project_member(output_root, root)
    return _create_bound_directory(bound_root_path / run_id, root=root)


def _feature_directory_from_freeze(freeze: Mapping[str, Any], *, root: Path) -> Path:
    records = freeze.get("feature_files")
    if not isinstance(records, dict) or set(records) != set(FEATURE_FILES):
        raise ValidationError("frozen feature inventory is incomplete")
    paths = {
        label: _verify_record(record, root=root, label=f"feature_files.{label}")
        for label, record in records.items()
    }
    parents = {path.parent for path in paths.values()}
    if len(parents) != 1:
        raise ValidationError("frozen feature files do not share one directory")
    directory = parents.pop()
    for label, filename in FEATURE_FILES.items():
        if paths[label] != directory / filename:
            raise ValidationError(f"frozen feature filename is not canonical: {label}")
    return directory


def _copy_outcome_snapshot(
    source: Path,
    destination: Path,
    expected: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Copy and validate the outcome after the prediction lock exists.

    This function is intentionally the first helper in the analysis path that
    probes the local outcome.  The private snapshot closes the pathname replacement window
    between publisher-integrity validation and parsing.
    """
    if source.name != str(expected["name"]):
        raise ValidationError(f"outcome basename must be exactly {expected['name']}")
    md5_digest = hashlib.md5(usedforsecurity=False)
    sha256_digest = hashlib.sha256()
    copied_bytes = 0

    def copy_and_validate(destination_handle: Any) -> None:
        nonlocal copied_bytes
        with source.open("rb") as source_handle:
            source_before = os.fstat(source_handle.fileno())
            if not stat_module.S_ISREG(source_before.st_mode):
                raise ValidationError("the local outcome source is not a regular file")
            while chunk := source_handle.read(4 * 1024 * 1024):
                md5_digest.update(chunk)
                sha256_digest.update(chunk)
                copied_bytes += len(chunk)
                destination_handle.write(chunk)
            source_after = os.fstat(source_handle.fileno())
            source_current = source.stat()
        if _stat_signature(source_before) != _stat_signature(
            source_after
        ) or _stat_signature(source_after) != _stat_signature(source_current):
            raise ValidationError("the local outcome file changed while it was copied")
        if copied_bytes != int(expected["bytes"]):
            raise ValidationError(
                "outcome byte count does not match frozen publisher metadata"
            )
        if md5_digest.hexdigest() != str(expected["md5"]):
            raise ValidationError(
                "outcome MD5 does not match frozen publisher metadata"
            )

    try:
        _write_once_artifact(destination, copy_and_validate, root=root)
    except FileNotFoundError as exc:
        raise ValidationError(
            "the local outcome file is unavailable after predictions were locked"
        ) from exc
    return {
        "path": _portable_relative(destination, root),
        "bytes": copied_bytes,
        "sha256": sha256_digest.hexdigest(),
    }


def _load_prediction_artifact_bytes(payload: bytes) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            observed = set(archive.files)
            expected = {"sample_ids", "gene_ids", "training_gene_mean"}
            for condition in CONDITIONS:
                slug = condition.lower().replace("-", "_")
                expected.add(f"{slug}_aggregate")
                expected.update(f"{slug}_seed_{seed}" for seed in SEEDS)
            if observed != expected:
                raise ValidationError(
                    f"prediction arrays differ from the frozen inventory: {sorted(observed ^ expected)}"
                )
            arrays = {name: np.asarray(archive[name]) for name in sorted(observed)}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(
            "prediction archive cannot be read without pickle"
        ) from exc

    samples = _require_unique(arrays["sample_ids"].tolist(), "prediction sample_ID")
    genes = _require_unique(arrays["gene_ids"].tolist(), "prediction gene")
    expected_shape = (len(samples), len(genes))
    if arrays["training_gene_mean"].shape != (len(genes),):
        raise ValidationError("training gene-mean vector has the wrong shape")
    for name, values in arrays.items():
        if name in {"sample_ids", "gene_ids"}:
            if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
                raise ValidationError(
                    f"identity array is not a one-dimensional string vector: {name}"
                )
            continue
        if name != "training_gene_mean" and values.shape != expected_shape:
            raise ValidationError(f"prediction matrix has the wrong shape: {name}")
        if values.dtype.kind not in {"f", "i", "u"} or not np.isfinite(values).all():
            raise ValidationError(
                f"prediction array is non-numeric or non-finite: {name}"
            )
    for condition in CONDITIONS:
        slug = condition.lower().replace("-", "_")
        expected_aggregate = np.mean(
            np.stack([arrays[f"{slug}_seed_{seed}"] for seed in SEEDS], axis=0),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        if not np.array_equal(arrays[f"{slug}_aggregate"], expected_aggregate):
            raise ValidationError(
                f"stored seed aggregate is not the fixed arithmetic mean: {condition}"
            )
    return arrays


def _load_prediction_artifact(
    path: Path,
    *,
    expected_record: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, np.ndarray]:
    payload, observed_record = _stable_read_bytes(
        path,
        label="prediction archive",
        root=root,
    )
    if expected_record is not None:
        _assert_same_record(
            observed_record, expected_record, label="prediction archive"
        )
    return _load_prediction_artifact_bytes(payload)


def _validate_prediction_lock(
    path: Path,
    *,
    freeze: Mapping[str, Any],
    freeze_record: Mapping[str, Any],
    root: Path,
    expected_lock_record: Mapping[str, Any] | None = None,
    trusted_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    if trusted_sha256 is not None:
        trusted_sha256 = trusted_sha256.strip().lower()
        if not SHA256_RE.fullmatch(trusted_sha256):
            raise ValidationError(
                "trusted prediction-lock SHA-256 must be 64 lowercase hexadecimal characters"
            )
    lock, consumed_lock_record = _load_json_mapping_with_record(
        path,
        "prediction lock",
        root=root,
    )
    if expected_lock_record is not None:
        _assert_same_record(
            consumed_lock_record, expected_lock_record, label="prediction lock"
        )
    if trusted_sha256 is not None and consumed_lock_record["sha256"] != trusted_sha256:
        raise ValidationError(
            "prediction lock does not match the independently supplied trusted SHA-256"
        )
    return _validate_prediction_lock_payload(
        lock,
        lock_parent=path.resolve(strict=True).parent,
        freeze=freeze,
        freeze_record=freeze_record,
        root=root,
    )


def _validate_prediction_lock_payload(
    lock: Mapping[str, Any],
    *,
    lock_parent: Path,
    freeze: Mapping[str, Any],
    freeze_record: Mapping[str, Any],
    root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Validate persisted prediction inputs before publishing their marker."""
    if (
        lock.get("schema") != PREDICTION_LOCK_SCHEMA
        or lock.get("numeric_outcome_argument_accepted_by_this_phase") is not False
        or lock.get("numeric_outcome_loaded_by_this_phase") is not False
    ):
        raise ValidationError("prediction lock has an invalid identity or phase state")
    if lock.get("seed_aggregation_complete") is not True:
        raise ValidationError(
            "prediction lock does not contain the complete seed aggregation"
        )
    if lock.get("freeze") != freeze_record:
        raise ValidationError(
            "prediction lock is not bound to the validated protocol lock"
        )
    prediction_record = lock.get("predictions", {})
    training_record = lock.get("training_records", {})
    if not isinstance(prediction_record, Mapping) or not isinstance(
        training_record, Mapping
    ):
        raise ValidationError("prediction lock artifact records are invalid")
    prediction_path = _safe_project_path(prediction_record.get("path"), root)
    training_path = _safe_project_path(training_record.get("path"), root)
    if prediction_path.parent != lock_parent.resolve(
        strict=True
    ) or training_path.parent != lock_parent.resolve(strict=True):
        raise ValidationError("prediction lock points outside its run directory")
    if (
        prediction_path.name != "predictions.npz"
        or training_path.name != "training-records.json"
    ):
        raise ValidationError("prediction lock artifact filenames are not canonical")
    arrays = _load_prediction_artifact(
        prediction_path,
        expected_record=prediction_record,
        root=root,
    )
    observed_metadata = {
        name: {"dtype": str(array.dtype), "shape": list(array.shape)}
        for name, array in arrays.items()
    }
    if lock.get("arrays") != observed_metadata:
        raise ValidationError(
            "prediction lock array metadata does not match the archive"
        )
    expected_genes, expected_samples, expected_axis_bindings = (
        _reconstruct_prediction_axes(freeze, root=root)
    )
    observed_genes = [str(value) for value in arrays["gene_ids"].tolist()]
    observed_samples = [str(value) for value in arrays["sample_ids"].tolist()]
    if observed_genes != expected_genes:
        raise ValidationError(
            "prediction gene order differs from frozen genes.txt line order"
        )
    if observed_samples != expected_samples:
        raise ValidationError(
            "prediction sample order or membership differs from the complete frozen eligible cohort"
        )
    if lock.get("prediction_axis_bindings") != expected_axis_bindings:
        raise ValidationError(
            "prediction lock axis bindings differ from frozen feature inputs"
        )
    training, consumed_training_record = _load_json_mapping_with_record(
        training_path,
        "training records",
        root=root,
    )
    _assert_same_record(
        consumed_training_record, training_record, label="training records"
    )
    if training.get("schema") != TRAINING_RECORDS_SCHEMA:
        raise ValidationError("unexpected full-refit record schema")
    if (
        training.get("numeric_outcome_argument_accepted_by_this_phase") is not False
        or training.get("numeric_outcome_loaded_by_this_phase") is not False
        or training.get("broad_training_cells") != 1103
    ):
        raise ValidationError(
            "full-refit record violates the frozen training/outcome boundary"
        )
    if (
        training.get("consumed_training_inputs") != freeze["training_inputs"]
        or training.get("private_training_snapshot_removed_after_load") is not True
    ):
        raise ValidationError(
            "full-refit record does not prove use of the frozen private input snapshot"
        )
    expected_executed_sources = _frozen_trusted_module_source_records(
        freeze, TRAINING_LOCAL_MODULES
    )
    if training.get("executed_local_module_sources") != expected_executed_sources:
        raise ValidationError(
            "full-refit executed module sources differ from the protocol lock"
        )
    gene_effect_record = next(
        (
            record
            for record in freeze["training_inputs"]
            if isinstance(record, Mapping)
            and record.get("path") == "data/processed/gene_effect.npy"
        ),
        None,
    )
    if gene_effect_record is None:
        raise ValidationError("protocol lock lacks the Broad training outcome matrix")
    gene_effect_path = _verify_record(
        gene_effect_record, root=root, label="Broad training outcomes"
    )
    gene_effect_bytes, consumed_gene_effect_record = _stable_read_bytes(
        gene_effect_path,
        label="Broad training outcomes",
        root=root,
    )
    _assert_same_record(
        consumed_gene_effect_record,
        gene_effect_record,
        label="Broad training outcomes",
    )
    try:
        gene_effect = np.load(io.BytesIO(gene_effect_bytes), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValidationError("Broad training outcome matrix cannot be loaded") from exc
    expected_gene_mean, expected_outcome_audit = _summarize_broad_training_outcomes(
        gene_effect
    )
    if not np.array_equal(arrays["training_gene_mean"], expected_gene_mean):
        raise ValidationError(
            "training gene mean does not match observed Broad targets"
        )
    if training.get("broad_training_outcomes") != expected_outcome_audit:
        raise ValidationError(
            "full-refit missing-target audit does not match Broad outcomes"
        )
    samples = [str(value) for value in arrays["sample_ids"].tolist()]
    feature_processing = training.get("external_feature_processing")
    if (
        training.get("external_prediction_samples") != len(samples)
        or not isinstance(feature_processing, Mapping)
        or feature_processing.get("sample_ids") != samples
        or training.get("prediction_axis_bindings") != expected_axis_bindings
    ):
        raise ValidationError(
            "full-refit records do not match the locked external prediction cohort"
        )
    training_device = training.get("device")
    if training_device not in {"cpu", "cuda"}:
        raise ValidationError("full-refit runtime metadata is incomplete")
    _validate_training_runtime(
        training.get("runtime"), expected_device=str(training_device)
    )
    model_rows = training.get("models")
    expected_keys = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if (
        not isinstance(model_rows, list)
        or len(model_rows) != len(expected_keys)
        or not all(isinstance(row, dict) for row in model_rows)
        or {(row.get("condition"), row.get("training_seed")) for row in model_rows}
        != expected_keys
    ):
        raise ValidationError(
            "full-refit records do not contain the six fixed model/seed fits"
        )
    expected_protocol_source = _frozen_trusted_module_source_records(
        freeze, ("corrected_protocol",)
    )[0]
    protocol = _load_trusted_module(
        "corrected_protocol",
        expected_source_record=expected_protocol_source,
        root=root,
    )
    _validate_loaded_local_modules(("corrected_protocol",), root=root)
    frozen_plans = {
        (row["condition"], int(row["training_seed"])): row
        for row in freeze["training_plan"]["plans"]
    }
    for row in model_rows:
        key = (row["condition"], int(row["training_seed"]))
        plan = frozen_plans.get(key)
        if (
            plan is None
            or row.get("config") != plan["selected_config"]
            or row.get("epochs") != plan["selected_epoch"]
        ):
            raise ValidationError(
                f"full-refit record differs from the frozen P1-derived plan: {key}"
            )
        expected_seed = protocol.derive_seed(key[1], "organoid-full-refit", key[0])
        training_detail = row.get("training")
        preprocessing = row.get("preprocessing")
        selected_epoch = int(plan["selected_epoch"])
        if row.get("full_refit_seed") != expected_seed:
            raise ValidationError(
                f"full-refit seed differs from the fixed derivation: {key}"
            )
        runtime_before_fit = row.get("runtime_before_fit")
        runtime_after_prediction = row.get("runtime_after_prediction")
        _validate_training_runtime(
            runtime_before_fit, expected_device=str(training_device)
        )
        _validate_training_runtime(
            runtime_after_prediction, expected_device=str(training_device)
        )
        if runtime_before_fit != runtime_after_prediction:
            raise ValidationError(
                f"full-refit runtime changed during model fit/prediction: {key}"
            )
        if (
            not isinstance(training_detail, Mapping)
            or training_detail.get("trained_epochs") != selected_epoch
            or training_detail.get("scheduler") != "CosineAnnealingLR"
            or training_detail.get("scheduler_horizon_epochs")
            != freeze["training_plan"]["scheduler_horizon"]
            or not isinstance(training_detail.get("history"), list)
            or len(training_detail["history"]) != selected_epoch
        ):
            raise ValidationError(
                f"full-refit training trace differs from the frozen plan: {key}"
            )
        if (
            not isinstance(preprocessing, Mapping)
            or preprocessing.get("n_fit_cells") != 1103
        ):
            raise ValidationError(
                f"full-refit preprocessing was not fit on all Broad cells: {key}"
            )
    return dict(lock), arrays, training


def _tissue_mapping_from_features(
    features_dir: Path,
    *,
    covariate_record: Mapping[str, Any],
    root: Path,
) -> dict[str, str]:
    covariate_bytes, consumed_record = _stable_read_bytes(
        features_dir / FEATURE_FILES["covariates"],
        label="organoid covariates",
        root=root,
    )
    _assert_same_record(consumed_record, covariate_record, label="organoid covariates")
    covariates = pd.read_csv(io.BytesIO(covariate_bytes), dtype={"sample_ID": str})
    sample_ids = _require_unique(covariates["sample_ID"], "covariate sample_ID")
    tissues = covariates["tissue"].astype(str).tolist()
    if set(tissues) - set(LINEAGE_MAP):
        raise ValidationError("covariates contain a tissue outside the fixed mapping")
    return dict(zip(sample_ids, tissues))


def lock_predictions(
    *,
    freeze_path: Path = DEFAULT_FREEZE,
    trusted_freeze_sha256: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    device: str = "auto",
    root: Path = ROOT,
) -> dict[str, Any]:
    """Train and lock predictions without accepting an outcome path."""
    with _lock_predictions_single_flight():
        if device not in {"auto", "cpu", "cuda"}:
            raise ValidationError("device must be one of: auto, cpu, cuda")
        _execution_source_records()
        freeze, freeze_record = validate_freeze(
            freeze_path,
            root=root,
            trusted_sha256=trusted_freeze_sha256,
        )
        _runtime_dependency_preflight()
        features_dir = _feature_directory_from_freeze(freeze, root=root)
        run_dir = _new_run_directory(output_root, run_id, root=root)
        _, _, prediction_lock = train_and_lock_predictions(
            freeze=freeze,
            freeze_record=freeze_record,
            run_dir=run_dir,
            features_dir=features_dir,
            device=device,
            root=root,
        )
        prediction_lock_path = run_dir / "prediction-lock.json"
        if prediction_lock_path.exists():
            raise ValidationError(
                "prediction training published its completion marker prematurely"
            )
        _validate_prediction_lock_payload(
            prediction_lock,
            lock_parent=run_dir,
            freeze=freeze,
            freeze_record=freeze_record,
            root=root,
        )
        freeze_after_training, freeze_record_after_training = validate_freeze(
            freeze_path,
            root=root,
            trusted_sha256=trusted_freeze_sha256,
            expected_record=freeze_record,
        )
        if (
            freeze_after_training != freeze
            or freeze_record_after_training != freeze_record
        ):
            raise ValidationError(
                "protocol lock changed during the prediction-lock phase"
            )
        for key, label in (
            ("predictions", "prediction archive"),
            ("training_records", "training records"),
        ):
            expected = prediction_lock.get(key)
            if not isinstance(expected, Mapping):
                raise ValidationError(f"{label} record is missing")
            observed = _record_file(
                _safe_project_path(expected.get("path"), root), root
            )
            _assert_same_record(observed, expected, label=label)
        _commit_execution_bound_json(
            prediction_lock_path,
            prediction_lock,
            root=root,
        )
        prediction_lock_record = _record_file(prediction_lock_path, root)
        return {
            "run_directory": _portable_relative(run_dir, root),
            "prediction_lock": prediction_lock_record,
            "trusted_prediction_lock_sha256": prediction_lock_record["sha256"],
            "external_retention_requirement": "Retain the freeze and prediction-lock SHA-256 values outside the run directory before starting analyze; local mutable receipts do not independently establish chronology.",
        }


def analyze_outcome(
    *,
    freeze_path: Path = DEFAULT_FREEZE,
    trusted_freeze_sha256: str,
    run_dir: Path,
    trusted_prediction_lock_sha256: str,
    outcome_path: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Analyze an outcome using a separately retained prediction-lock digest."""
    trusted_freeze = trusted_freeze_sha256.strip().lower()
    trusted_prediction = trusted_prediction_lock_sha256.strip().lower()
    if not SHA256_RE.fullmatch(trusted_freeze):
        raise ValidationError(
            "trusted freeze SHA-256 must be 64 lowercase hexadecimal characters"
        )
    if not SHA256_RE.fullmatch(trusted_prediction):
        raise ValidationError(
            "trusted prediction-lock SHA-256 must be 64 lowercase hexadecimal characters"
        )
    _execution_source_records()
    prevalidated_freeze, prevalidated_freeze_record = _prevalidate_freeze_trust(
        freeze_path,
        trusted_sha256=trusted_freeze,
        root=root,
    )
    run_dir = run_dir.resolve(strict=True)
    if not run_dir.is_dir() or not _path_within(run_dir, root):
        raise ValidationError(
            "run directory must be an existing directory inside the project root"
        )
    prediction_phase_files = {
        "predictions.npz",
        "training-records.json",
        "prediction-lock.json",
    }
    observed_entries = {path.name for path in run_dir.iterdir()}
    if observed_entries != prediction_phase_files:
        raise ValidationError(
            "run directory is not an exact incomplete prediction-lock phase"
        )
    prediction_lock_path = run_dir / "prediction-lock.json"
    _, prediction_lock_record = _prevalidate_prediction_lock_trust(
        prediction_lock_path,
        trusted_sha256=trusted_prediction,
        freeze_record=prevalidated_freeze_record,
        root=root,
    )
    _load_scientific_runtime()
    freeze, freeze_record = validate_freeze(
        freeze_path,
        root=root,
        trusted_sha256=trusted_freeze,
        expected_record=prevalidated_freeze_record,
    )
    if freeze != prevalidated_freeze:
        raise ValidationError("protocol lock changed after dependency loading")
    features_dir = _feature_directory_from_freeze(freeze, root=root)
    _, predictions, _ = _validate_prediction_lock(
        prediction_lock_path,
        freeze=freeze,
        freeze_record=freeze_record,
        root=root,
        expected_lock_record=prediction_lock_record,
        trusted_sha256=trusted_prediction,
    )
    freeze_before_outcome, freeze_record_before_outcome = validate_freeze(
        freeze_path,
        root=root,
        trusted_sha256=trusted_freeze,
        expected_record=freeze_record,
    )
    if freeze_before_outcome != freeze or freeze_record_before_outcome != freeze_record:
        raise ValidationError("protocol lock changed before outcome analysis")
    if _record_file(prediction_lock_path, root) != prediction_lock_record:
        raise ValidationError("prediction lock changed before outcome analysis")

    expected_outcome = freeze["figshare"]["publisher_outcome_files"][
        "primary_qnorm_lfc"
    ]
    outcome_snapshot = run_dir / "outcome-snapshot.RData"
    outcome_record = _copy_outcome_snapshot(
        outcome_path, outcome_snapshot, expected_outcome, root=root
    )
    prediction_samples = [str(value) for value in predictions["sample_ids"].tolist()]
    outcome = _load_bound_outcome(
        outcome_snapshot,
        expected_record=outcome_record,
        publisher_metadata=expected_outcome,
        allowed_samples=prediction_samples,
        root=root,
    )
    tissue_by_sample = _tissue_mapping_from_features(
        features_dir,
        covariate_record=freeze["feature_files"]["covariates"],
        root=root,
    )
    analysis, cell_rows, gene_rows = compute_analysis(
        predictions,
        outcome,
        tissue_by_sample=tissue_by_sample,
    )

    _atomic_json(run_dir / "analysis.json", analysis, root=root)
    _atomic_tsv(run_dir / "cellwise-metrics.tsv", cell_rows, CELLWISE_FIELDS, root=root)
    _atomic_tsv(
        run_dir / "per-gene-correlations.tsv", gene_rows, GENE_FIELDS, root=root
    )
    bindings = {
        "schema": SOURCE_BINDINGS_SCHEMA,
        "created_utc": _utc_now(),
        "freeze": dict(freeze_record),
        "trusted_freeze_sha256_supplied_to_analysis": trusted_freeze,
        "prediction_lock": dict(prediction_lock_record),
        "trusted_prediction_lock_sha256_supplied_to_analysis": trusted_prediction,
        "outcome_snapshot": dict(outcome_record),
        "outcome_publisher_metadata": dict(expected_outcome),
        "integrity_scope": "external digest matching plus deterministic analysis replay from locked predictions and artifact validation; these records do not independently establish real-world chronology",
        "absolute_outcome_source_path_recorded": False,
    }
    _atomic_json(run_dir / "source-bindings.json", bindings, root=root)

    if _record_file(prediction_lock_path, root) != prediction_lock_record:
        raise ValidationError("prediction lock changed during outcome analysis")
    if _record_file(outcome_snapshot, root) != outcome_record:
        raise ValidationError("outcome snapshot changed during analysis")
    freeze_final, freeze_record_final = validate_freeze(
        freeze_path,
        root=root,
        trusted_sha256=trusted_freeze,
        expected_record=freeze_record,
    )
    if freeze_final != freeze or freeze_record_final != freeze_record:
        raise ValidationError("protocol lock changed during analysis")

    artifact_names = (
        "predictions.npz",
        "training-records.json",
        "prediction-lock.json",
        "outcome-snapshot.RData",
        "analysis.json",
        "cellwise-metrics.tsv",
        "per-gene-correlations.tsv",
        "source-bindings.json",
    )
    artifacts = {name: _record_file(run_dir / name, root) for name in artifact_names}
    completion = {
        "schema": COMPLETION_SCHEMA,
        "created_utc": _utc_now(),
        "scientific_result": True,
        "run_id": run_dir.name,
        "freeze": dict(freeze_record),
        "trusted_freeze_sha256": trusted_freeze,
        "trusted_prediction_lock_sha256": trusted_prediction,
        "outcome_snapshot_sha256": outcome_record["sha256"],
        "analysis_canonical_sha256": hashlib.sha256(
            _canonical_json_bytes(analysis)
        ).hexdigest(),
        "artifacts": artifacts,
        "primary_results": analysis["primary_endpoint"]["models"],
        "conclusion": analysis["conclusion"],
        "verification_requirement": "Supply the externally retained freeze, prediction-lock, and completion SHA-256 values to verify.",
    }
    completion_path = run_dir / "completion.json"
    _commit_execution_bound_json(completion_path, completion, root=root)
    completion_sha256 = _sha256_file(completion_path)
    return {
        "run_directory": _portable_relative(run_dir, root),
        "completion": _record_file(completion_path, root),
        "trusted_completion_sha256": completion_sha256,
        "primary_results": completion["primary_results"],
        "conclusion": completion["conclusion"],
    }


def verify_run(
    *,
    run_dir: Path,
    trusted_freeze_sha256: str,
    trusted_prediction_lock_sha256: str,
    trusted_completion_sha256: str,
    output: Path | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Replay analysis from locked predictions and check artifact integrity."""
    trusted_freeze = trusted_freeze_sha256.strip().lower()
    trusted_prediction = trusted_prediction_lock_sha256.strip().lower()
    trusted_completion = trusted_completion_sha256.strip().lower()
    for label, digest in (
        ("freeze", trusted_freeze),
        ("prediction-lock", trusted_prediction),
        ("completion", trusted_completion),
    ):
        if not SHA256_RE.fullmatch(digest):
            raise ValidationError(
                f"trusted {label} SHA-256 must be 64 lowercase hexadecimal characters"
            )
    _execution_source_records()
    run_dir = run_dir.resolve(strict=True)
    if not run_dir.is_dir() or not _path_within(run_dir, root):
        raise ValidationError(
            "run directory must be an existing directory inside the project root"
        )
    completion_path = (run_dir / "completion.json").resolve(strict=True)
    if not completion_path.is_file() or completion_path.parent != run_dir:
        raise ValidationError("completion record escapes or is not a regular file")
    completion_bytes, consumed_completion_record = _stable_read_bytes(
        completion_path,
        label="completion record",
        root=root,
    )
    if consumed_completion_record["sha256"] != trusted_completion:
        raise ValidationError(
            "completion.json does not match the independently supplied trusted SHA-256"
        )
    completion = _parse_json_mapping(completion_bytes, "completion record")
    if (
        completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("scientific_result") is not True
    ):
        raise ValidationError(
            "completion record has an invalid schema or scientific-result state"
        )
    if completion.get("run_id") != run_dir.name:
        raise ValidationError("completion run ID does not match its directory")
    artifact_records = completion.get("artifacts")
    expected_artifacts = {
        "predictions.npz",
        "training-records.json",
        "prediction-lock.json",
        "outcome-snapshot.RData",
        "analysis.json",
        "cellwise-metrics.tsv",
        "per-gene-correlations.tsv",
        "source-bindings.json",
    }
    if (
        not isinstance(artifact_records, dict)
        or set(artifact_records) != expected_artifacts
    ):
        raise ValidationError(
            "completion artifact inventory is incomplete or has extra entries"
        )
    for name, record in artifact_records.items():
        fields = _record_fields(record, f"completion.artifacts.{name}")
        if (
            not isinstance(fields["path"], str)
            or not isinstance(fields["bytes"], int)
            or isinstance(fields["bytes"], bool)
            or fields["bytes"] < 0
            or not isinstance(fields["sha256"], str)
            or not SHA256_RE.fullmatch(fields["sha256"])
        ):
            raise ValidationError(f"completion.artifacts.{name} file record is invalid")

    freeze_record_from_completion = completion.get("freeze")
    if not isinstance(freeze_record_from_completion, dict):
        raise ValidationError("completion protocol-lock record is invalid")
    _record_fields(freeze_record_from_completion, "completion.freeze")
    if completion.get("trusted_freeze_sha256") != trusted_freeze:
        raise ValidationError(
            "completion does not match the externally supplied trusted freeze SHA-256"
        )
    if freeze_record_from_completion.get("sha256") != trusted_freeze:
        raise ValidationError("frozen record does not match the trusted freeze SHA-256")

    prediction_lock_record = artifact_records["prediction-lock.json"]
    if completion.get("trusted_prediction_lock_sha256") != trusted_prediction:
        raise ValidationError(
            "completion does not match the externally supplied prediction-lock SHA-256"
        )
    if prediction_lock_record.get("sha256") != trusted_prediction:
        raise ValidationError(
            "prediction-lock artifact does not match the trusted SHA-256"
        )

    artifact_paths: dict[str, Path] = {}
    for name, record in artifact_records.items():
        if name == "outcome-snapshot.RData":
            continue
        path = _verify_record(record, root=root, label=f"completion.artifacts.{name}")
        if path.parent != run_dir or path.name != name:
            raise ValidationError(f"completion artifact escapes or is renamed: {name}")
        artifact_paths[name] = path

    bindings_bytes, consumed_bindings_record = _stable_read_bytes(
        artifact_paths["source-bindings.json"],
        label="source bindings",
        root=root,
    )
    _assert_same_record(
        consumed_bindings_record,
        artifact_records["source-bindings.json"],
        label="source bindings",
    )
    bindings = _parse_json_mapping(bindings_bytes, "source bindings")
    if bindings.get("schema") != SOURCE_BINDINGS_SCHEMA:
        raise ValidationError("source bindings have an unexpected schema")
    if bindings.get("absolute_outcome_source_path_recorded") is not False:
        raise ValidationError(
            "source bindings unexpectedly contain an absolute outcome path"
        )
    freeze_record = bindings.get("freeze")
    if (
        not isinstance(freeze_record, dict)
        or freeze_record_from_completion != freeze_record
    ):
        raise ValidationError(
            "completion and source bindings disagree on the protocol lock"
        )
    freeze_path = _verify_record(
        freeze_record, root=root, label="source_bindings.freeze"
    )
    if bindings.get("trusted_freeze_sha256_supplied_to_analysis") != trusted_freeze:
        raise ValidationError("source bindings do not match the trusted freeze SHA-256")

    if bindings.get("prediction_lock") != prediction_lock_record:
        raise ValidationError(
            "source bindings do not match the prediction-lock artifact"
        )
    if (
        bindings.get("trusted_prediction_lock_sha256_supplied_to_analysis")
        != trusted_prediction
    ):
        raise ValidationError(
            "source bindings do not match the trusted prediction-lock SHA-256"
        )

    outcome_record = artifact_records["outcome-snapshot.RData"]
    if bindings.get("outcome_snapshot") != outcome_record:
        raise ValidationError("source bindings do not match the outcome snapshot")
    if completion.get("outcome_snapshot_sha256") != outcome_record["sha256"]:
        raise ValidationError("completion does not match the outcome snapshot digest")
    outcome_path = _verify_record(
        outcome_record,
        root=root,
        label="completion.artifacts.outcome-snapshot.RData",
    )
    if outcome_path.parent != run_dir or outcome_path.name != "outcome-snapshot.RData":
        raise ValidationError(
            "completion artifact escapes or is renamed: outcome-snapshot.RData"
        )
    artifact_paths["outcome-snapshot.RData"] = outcome_path

    _load_scientific_runtime()
    freeze, verified_freeze_record = validate_freeze(
        freeze_path,
        root=root,
        trusted_sha256=trusted_freeze,
        expected_record=freeze_record,
    )
    if verified_freeze_record != freeze_record:
        raise ValidationError(
            "current protocol-lock record differs from the run binding"
        )
    features_dir = _feature_directory_from_freeze(freeze, root=root)

    _, predictions, _ = _validate_prediction_lock(
        artifact_paths["prediction-lock.json"],
        freeze=freeze,
        freeze_record=freeze_record,
        root=root,
        expected_lock_record=prediction_lock_record,
        trusted_sha256=trusted_prediction,
    )

    expected_outcome = freeze["figshare"]["publisher_outcome_files"][
        "primary_qnorm_lfc"
    ]
    if bindings.get("outcome_publisher_metadata") != expected_outcome:
        raise ValidationError(
            "source bindings do not match frozen publisher outcome metadata"
        )
    prediction_samples = [str(value) for value in predictions["sample_ids"].tolist()]
    outcome = _load_bound_outcome(
        outcome_path,
        expected_record=outcome_record,
        publisher_metadata=expected_outcome,
        allowed_samples=prediction_samples,
        root=root,
    )
    tissue_by_sample = _tissue_mapping_from_features(
        features_dir,
        covariate_record=freeze["feature_files"]["covariates"],
        root=root,
    )
    analysis, cell_rows, gene_rows = compute_analysis(
        predictions,
        outcome,
        tissue_by_sample=tissue_by_sample,
    )
    analysis_bytes, consumed_analysis_record = _stable_read_bytes(
        artifact_paths["analysis.json"],
        label="analysis record",
        root=root,
    )
    _assert_same_record(
        consumed_analysis_record,
        artifact_records["analysis.json"],
        label="analysis record",
    )
    stored_analysis = _parse_json_mapping(analysis_bytes, "analysis record")
    if _canonical_json_bytes(stored_analysis) != _canonical_json_bytes(analysis):
        raise ValidationError(
            "stored analysis does not exactly match deterministic recomputation"
        )
    if (
        completion.get("analysis_canonical_sha256")
        != hashlib.sha256(_canonical_json_bytes(analysis)).hexdigest()
    ):
        raise ValidationError(
            "completion analysis digest does not match deterministic recomputation"
        )
    if completion.get("primary_results") != analysis["primary_endpoint"]["models"]:
        raise ValidationError("completion primary results do not match recomputation")
    if completion.get("conclusion") != analysis["conclusion"]:
        raise ValidationError("completion conclusion does not match recomputation")
    cellwise_bytes, consumed_cellwise_record = _stable_read_bytes(
        artifact_paths["cellwise-metrics.tsv"],
        label="cellwise metrics",
        root=root,
    )
    _assert_same_record(
        consumed_cellwise_record,
        artifact_records["cellwise-metrics.tsv"],
        label="cellwise metrics",
    )
    if cellwise_bytes != _tsv_bytes(cell_rows, CELLWISE_FIELDS):
        raise ValidationError(
            "cellwise table does not exactly match deterministic recomputation"
        )
    gene_bytes, consumed_gene_record = _stable_read_bytes(
        artifact_paths["per-gene-correlations.tsv"],
        label="per-gene correlations",
        root=root,
    )
    _assert_same_record(
        consumed_gene_record,
        artifact_records["per-gene-correlations.tsv"],
        label="per-gene correlations",
    )
    if gene_bytes != _tsv_bytes(gene_rows, GENE_FIELDS):
        raise ValidationError(
            "per-gene table does not exactly match deterministic recomputation"
        )

    report = {
        "schema": VERIFICATION_SCHEMA,
        "created_utc": _utc_now(),
        "scientific_result": False,
        "verification_passed": True,
        "verification_scope": "deterministic analysis replay from locked predictions plus artifact-integrity verification; no model retraining or prediction regeneration and not independent scientific verification",
        "run_directory": _portable_relative(run_dir, root),
        "trusted_freeze_sha256": trusted_freeze,
        "trusted_prediction_lock_sha256": trusted_prediction,
        "trusted_completion_sha256": trusted_completion,
        "completion": consumed_completion_record,
        "analysis_canonical_sha256": completion["analysis_canonical_sha256"],
        "recomputed_primary_results": analysis["primary_endpoint"]["models"],
        "recomputed_conclusion": analysis["conclusion"],
    }
    if output is not None:
        output_resolved = _project_output_path(output, root=root)
        if _path_within(output_resolved, run_dir):
            raise ValidationError(
                "verification output must be outside the immutable run directory"
            )
        _commit_execution_bound_json(output_resolved, report, root=root)
    _execution_source_records()
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-protocol Sanger organoid validation workflow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit-features", help="audit only non-outcome external features"
    )
    audit.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    audit.add_argument("--genes", type=Path, default=DEFAULT_GENES)
    audit.add_argument("--output", type=Path, required=True)

    freeze = subparsers.add_parser("freeze", help="write the fixed-protocol lock once")
    freeze.add_argument("--output", type=Path, default=DEFAULT_FREEZE)
    freeze.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    freeze.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    freeze.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)

    lock = subparsers.add_parser(
        "lock-predictions",
        help="train and lock predictions without accepting an outcome path",
    )
    lock.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    lock.add_argument("--trusted-freeze-sha256", required=True)
    lock.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    lock.add_argument("--run-id")
    lock.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    analyze = subparsers.add_parser(
        "analyze",
        help="analyze an outcome against externally retained freeze and prediction-lock digests",
    )
    analyze.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    analyze.add_argument("--trusted-freeze-sha256", required=True)
    analyze.add_argument("--run-dir", type=Path, required=True)
    analyze.add_argument("--trusted-prediction-lock-sha256", required=True)
    analyze.add_argument("--outcome", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify",
        help="deterministic analysis replay from locked predictions against externally retained freeze, prediction-lock, and completion digests",
    )
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--trusted-freeze-sha256", required=True)
    verify.add_argument("--trusted-prediction-lock-sha256", required=True)
    verify.add_argument("--trusted-completion-sha256", required=True)
    verify.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "audit-features":
            report = audit_features(
                features_dir=args.features_dir,
                genes_path=args.genes,
                output=args.output,
            )
            summary = {
                "command": args.command,
                "audit_passed": report["audit_passed"],
                "output": _portable_relative(args.output),
                "numeric_outcome_loaded_by_this_phase": False,
            }
        elif args.command == "freeze":
            freeze = create_freeze(
                output=args.output,
                campaign_path=args.campaign,
                features_dir=args.features_dir,
                protocol_path=args.protocol,
            )
            summary = {
                "command": args.command,
                "output": _portable_relative(args.output),
                "lock_body_sha256": freeze["lock_body_sha256"],
                "trusted_freeze_sha256": _sha256_file(args.output),
                "numeric_outcome_loaded_by_this_phase": False,
            }
        elif args.command == "lock-predictions":
            summary = {
                "command": args.command,
                **lock_predictions(
                    freeze_path=args.freeze,
                    trusted_freeze_sha256=args.trusted_freeze_sha256,
                    output_root=args.output_root,
                    run_id=args.run_id,
                    device=args.device,
                ),
            }
        elif args.command == "analyze":
            summary = {
                "command": args.command,
                **analyze_outcome(
                    freeze_path=args.freeze,
                    trusted_freeze_sha256=args.trusted_freeze_sha256,
                    run_dir=args.run_dir,
                    trusted_prediction_lock_sha256=(
                        args.trusted_prediction_lock_sha256
                    ),
                    outcome_path=args.outcome,
                ),
            }
        else:
            summary = {
                "command": args.command,
                **verify_run(
                    run_dir=args.run_dir,
                    trusted_freeze_sha256=args.trusted_freeze_sha256,
                    trusted_prediction_lock_sha256=(
                        args.trusted_prediction_lock_sha256
                    ),
                    trusted_completion_sha256=args.trusted_completion_sha256,
                    output=args.output,
                ),
            }
    except (ValidationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            summary, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
        )
    )
    return 0


if __name__ == "__main__":
    print(
        "ERROR: invoke code/run_organoid_independent_validation.py so the "
        "controller and launcher are hash-bound before execution",
        file=sys.stderr,
    )
    raise SystemExit(2)
