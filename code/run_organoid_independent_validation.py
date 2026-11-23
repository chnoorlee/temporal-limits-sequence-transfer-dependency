"""Minimal source-binding launcher for the organoid validation controller."""

# ruff: noqa: E402

from __future__ import annotations

import sys as _bootstrap_sys

_FORMAL_INVOCATION = (
    r".\.venv\Scripts\python.exe -I code\run_organoid_independent_validation.py"
)


def _require_isolated_interpreter() -> None:
    """Reject an interpreter that may already have executed site customisation."""
    if _bootstrap_sys.flags.isolated:
        return
    raise SystemExit(
        "ERROR: organoid validation requires an isolated interpreter from process "
        f"creation; invoke: {_FORMAL_INVOCATION}"
    )


if __name__ == "__main__":
    _require_isolated_interpreter()

import hashlib
import os
import stat
import sys
import types
from builtins import BaseExceptionGroup
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NoReturn

_LAUNCHER_RELATIVE = "code/run_organoid_independent_validation.py"
_CONTROLLER_RELATIVE = "code/organoid_independent_validation.py"
_CONTROLLER_RECORD_KEY = "_ORGANOID_CONTROLLER_EXECUTED_SOURCE_RECORD"
_CONTROLLER_PATH_KEY = "_ORGANOID_CONTROLLER_EXECUTED_SOURCE_PATH"
_LAUNCHER_RECORD_KEY = "_ORGANOID_LAUNCHER_EXECUTED_SOURCE_RECORD"
_LAUNCHER_PATH_KEY = "_ORGANOID_LAUNCHER_EXECUTED_SOURCE_PATH"
_BOUND_MODULE_NAME = "_organoid_independent_validation_bound"
_LAUNCHER_SELF_SHA256 = "61dfb9feb3e56f18c86aab022193a81bc23172c1d634b3cb77284fcb8bf809ef"  # fmt: skip
_EXECUTED_LAUNCHER_CODE = sys._getframe().f_code


class LauncherError(RuntimeError):
    """Raised when executable source identity cannot be established."""


CleanupFailure = tuple[str, BaseException]


def _capture_cleanup_failure(
    failures: list[CleanupFailure],
    label: str,
    cleanup: Callable[[], None],
) -> None:
    try:
        cleanup()
    except BaseException as exc:
        failures.append((label, exc))


def _raise_after_cleanup(
    primary: BaseException, cleanup_failures: Sequence[CleanupFailure]
) -> NoReturn:
    for label, failure in cleanup_failures:
        primary.add_note(
            f"launcher cleanup failure [{label}]: "
            f"{type(failure).__name__}: {failure}"
        )
    distinct = [failure for _, failure in cleanup_failures if failure is not primary]
    if not distinct:
        raise primary.with_traceback(primary.__traceback__)
    if len(distinct) == 1:
        diagnostic: BaseException = distinct[0]
    else:
        diagnostic = BaseExceptionGroup(
            "launcher cleanup failures while preserving the primary failure",
            distinct,
        )
    raise primary.with_traceback(primary.__traceback__) from diagnostic


def _install_bound_controller(module: types.ModuleType) -> None:
    sys.modules[_BOUND_MODULE_NAME] = module


def _restore_bound_controller_registration(
    had_module: bool, previous_module: types.ModuleType | None
) -> None:
    if had_module:
        sys.modules[_BOUND_MODULE_NAME] = previous_module
    else:
        sys.modules.pop(_BOUND_MODULE_NAME, None)


def _signature(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _stable_read(path: Path, relative: str, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        resolved = path.resolve(strict=True)
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise LauncherError(f"{label} is not a regular file")
            payload = handle.read()
            after = os.fstat(handle.fileno())
        current = resolved.stat()
    except LauncherError:
        raise
    except OSError as exc:
        raise LauncherError(f"{label} cannot be read: {path}") from exc
    if (
        _signature(before) != _signature(after)
        or _signature(after) != _signature(current)
        or len(payload) != int(after.st_size)
    ):
        raise LauncherError(f"{label} changed while it was being read")
    return payload, {
        "path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _normalised_launcher_sha256(source: bytes) -> str:
    prefix = b"_LAUNCHER_SELF_" + b'SHA256 = "'
    start = source.find(prefix)
    if start < 0 or source.find(prefix, start + 1) >= 0:
        raise LauncherError("launcher self-hash declaration is not unique")
    digest_start = start + len(prefix)
    digest_end = digest_start + 64
    declared = source[digest_start:digest_end]
    if (
        len(declared) != 64
        or any(value not in b"0123456789abcdef" for value in declared)
        or source[digest_end : digest_end + 1] != b'"'
    ):
        raise LauncherError("launcher self-hash declaration is invalid")
    normalised = source[:digest_start] + (b"0" * 64) + source[digest_end:]
    return hashlib.sha256(normalised).hexdigest()


def _verified_launcher_record() -> tuple[Path, dict[str, Any]]:
    launcher_path = Path(__file__).resolve(strict=True)
    source, record = _stable_read(launcher_path, _LAUNCHER_RELATIVE, "launcher source")
    compiled = compile(
        source,
        str(launcher_path),
        "exec",
        dont_inherit=True,
        optimize=sys.flags.optimize,
    )
    if compiled != _EXECUTED_LAUNCHER_CODE:
        raise LauncherError("launcher source does not match the code being executed")
    if _normalised_launcher_sha256(source) != _LAUNCHER_SELF_SHA256:
        raise LauncherError("launcher source does not match its embedded self-hash")
    return launcher_path, record


def _assert_record_unchanged(
    path: Path, expected: dict[str, Any], relative: str, label: str
) -> None:
    _, observed = _stable_read(path, relative, label)
    if observed != expected:
        raise LauncherError(f"{label} changed after binding")


def load_controller() -> types.ModuleType:
    """Load one verified controller snapshot into a fresh module namespace."""
    launcher_path, launcher_record = _verified_launcher_record()
    controller_path = launcher_path.with_name("organoid_independent_validation.py")
    controller_source, controller_record = _stable_read(
        controller_path,
        _CONTROLLER_RELATIVE,
        "controller execution source",
    )
    controller_code = compile(
        controller_source,
        str(controller_path),
        "exec",
        dont_inherit=True,
        optimize=sys.flags.optimize,
    )
    had_module = _BOUND_MODULE_NAME in sys.modules
    previous_module = sys.modules.get(_BOUND_MODULE_NAME)
    if had_module:
        raise LauncherError("the bound controller namespace already exists")

    module = types.ModuleType(_BOUND_MODULE_NAME)
    namespace = module.__dict__
    namespace.update(
        {
            "__file__": str(controller_path),
            "__package__": None,
            "__cached__": None,
            "__loader__": None,
            "__spec__": None,
            _CONTROLLER_RECORD_KEY: types.MappingProxyType(dict(controller_record)),
            _CONTROLLER_PATH_KEY: controller_path,
            _LAUNCHER_RECORD_KEY: types.MappingProxyType(dict(launcher_record)),
            _LAUNCHER_PATH_KEY: launcher_path,
        }
    )
    primary_failure: BaseException | None = None
    cleanup_failures: list[CleanupFailure] = []
    try:
        _install_bound_controller(module)
        exec(controller_code, namespace)
        if sys.modules.get(_BOUND_MODULE_NAME) is not module:
            raise LauncherError("the bound controller replaced its module registration")
        _assert_record_unchanged(
            controller_path,
            controller_record,
            _CONTROLLER_RELATIVE,
            "controller execution source",
        )
        _assert_record_unchanged(
            launcher_path,
            launcher_record,
            _LAUNCHER_RELATIVE,
            "launcher execution source",
        )
    except BaseException as exc:
        primary_failure = exc

    if primary_failure is not None:
        for attempt in ("initial", "second-chance"):
            _capture_cleanup_failure(
                cleanup_failures,
                f"{attempt} bound-controller registration restoration",
                lambda: _restore_bound_controller_registration(
                    had_module, previous_module
                ),
            )
        _raise_after_cleanup(primary_failure, cleanup_failures)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    _require_isolated_interpreter()
    controller = load_controller()
    controller_main = getattr(controller, "main", None)
    if not callable(controller_main):
        raise LauncherError("bound controller does not expose main()")
    return int(controller_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
