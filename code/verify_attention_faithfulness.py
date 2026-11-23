"""Standalone-process verifier entry for attention-faithfulness evidence.

Campaign replay lazily imports ``attention_faithfulness`` and therefore uses
its PyTorch implementation. ``--verify-receipt`` is a standard-library-only
checksum/schema replay: it never imports or deserializes with PyTorch. Both
paths reuse the same evidence schema and are not independent implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(os.path.abspath(os.fspath(__file__))).parent.parent
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "attention_faithfulness_v1"
SCHEMA_VERSION = "DepPLM-attention-faithfulness-v1"
FILE_RECORD_FIELDS = frozenset({"path", "bytes", "sha256"})
RECEIPT_FIELDS = frozenset(
    {
        "schema", "created_utc", "source_output_root", "scope",
        "verification_passed", "scientific_result", "summary",
        "accepted_attempt_completion_roots", "verifier_code", "assurance_scope",
    }
)
COMPLETION_ARTIFACT_KEYS = frozenset(
    {"predictions", "attention_evidence", "inference", "metrics"}
)
COMPLETION_FIELDS = frozenset(
    {
        "schema", "completed_utc", "seed", "fold", "pilot", "run_status",
        "protocol_complete", "eligible_for_scientific_aggregation",
        "scientific_result", "scientific_result_reason", "training",
        "run_manifest", "artifacts", "prediction_generation_calls",
        "outer_test_prediction_bundle_generation_calls",
        "one_prediction_row_per_outer_test_cell_per_mode",
        "wall_seconds_after_training", "scope_statement",
    }
)
MANIFEST_FIELDS = frozenset(
    {
        "schema", "created_utc", "seed", "fold", "pilot", "scientific_result",
        "protocol", "condition", "outer_train_indices_sha256",
        "outer_test_indices_sha256", "arguments", "corrected_protocol_arguments",
        "perturbation_protocol", "code", "inputs", "environment", "run_status",
        "eligible_for_scientific_aggregation",
    }
)
MANIFEST_ARGUMENT_FIELDS = frozenset(
    {"device", "attention_batch_size", "gene_chunk_size", "pilot_cells", "pilot_genes"}
)
TRAINING_FIELDS = frozenset(
    {
        "schema", "seed", "fold", "fold_seed", "inner_train_indices_sha256",
        "inner_validation_indices_sha256", "outer_train_indices_sha256",
        "outer_test_indices_sha256", "fit_indices_sha256", "best_config",
        "best_epoch", "selection_score", "tuning", "refit",
        "train_gene_mean_sha256", "inner_preprocessing", "preprocessing",
        "checkpoint", "outer_test_labels_accessed_during_training_or_prediction",
        "scientific_result",
    }
)
INFERENCE_FIELDS = frozenset(
    {
        "schema", "prediction_generation_calls",
        "outer_test_prediction_bundle_generation_calls",
        "one_prediction_row_per_outer_test_cell_per_mode", "outer_test_rows",
        "outer_test_unique_rows",
        "outer_test_labels_accessed_during_prediction_generation",
        "prediction_modes", "mask_plans", "attention_permutation",
        "parameter_randomization", "predictions", "attention_evidence",
        "scope_statement",
    }
)
METRICS_FIELDS = frozenset(
    {
        "schema", "computed_only_after_predictions_and_evidence_were_atomically_persisted",
        "outer_test_target_sha256", "modes", "scope_statement", "scientific_result",
    }
)


class ReceiptVerificationError(RuntimeError):
    """Raised when standard-library-only receipt replay fails closed."""


def _load_faith_module() -> Any:
    return importlib.import_module("attention_faithfulness")


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_no_link_or_reparse(path: Path) -> None:
    """Reject observed symlink/reparse components before receipt replay."""

    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            raise ReceiptVerificationError(
                f"detached verification path is missing: {current}"
            )
        except OSError as exc:
            raise ReceiptVerificationError(
                f"cannot inspect detached verification path {current}: {exc}"
            ) from exc
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or attributes & reparse_flag:
            raise ReceiptVerificationError(
                "detached verification path contains a symlink or reparse "
                f"point: {current}"
            )


def _require_external_artifact(
    path: Path, *, output_root: Path, label: str
) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ReceiptVerificationError(f"cannot resolve {label}: {path}") from exc
    if _is_within(resolved, output_root):
        raise ReceiptVerificationError(f"{label} must be outside output_root")
    return resolved


def _resolve_path(path: Path, *, label: str, strict: bool = False) -> Path:
    try:
        return path.resolve(strict=strict)
    except (OSError, RuntimeError) as exc:
        raise ReceiptVerificationError(f"cannot resolve {label}: {path}") from exc


def _validate_record(record: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(record, Mapping) or set(record) != FILE_RECORD_FIELDS:
        raise ReceiptVerificationError(f"{label} is not an exact file record")
    path, size, digest = record.get("path"), record.get("bytes"), record.get("sha256")
    if not isinstance(path, str) or not path:
        raise ReceiptVerificationError(f"{label} has an invalid path")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ReceiptVerificationError(f"{label} has an invalid byte count")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ReceiptVerificationError(f"{label} has an invalid SHA256 digest")
    return dict(record)


def _resolve_record(record: Mapping[str, Any], *, root: Path, label: str) -> Path:
    root = _resolve_path(root, label=f"{label} root", strict=True)
    recorded = Path(str(record["path"]))
    resolved = _resolve_path(
        recorded if recorded.is_absolute() else root / recorded,
        label=f"{label} recorded path",
    )
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReceiptVerificationError(f"{label} resolves outside its root") from exc
    return resolved


def _snapshot_record(
    record: Any, *, root: Path, label: str, expected: Path | None = None
) -> tuple[Path, bytes]:
    normalized = _validate_record(record, label=label)
    path = _resolve_record(normalized, root=root, label=label)
    if expected is not None:
        expected_resolved = _resolve_path(expected, label=f"{label} expected path")
        if path != expected_resolved:
            raise ReceiptVerificationError(f"{label} path mismatch")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ReceiptVerificationError(f"recorded artifact is missing: {path}") from exc
    if len(data) != normalized["bytes"] or hashlib.sha256(data).hexdigest() != normalized["sha256"]:
        raise ReceiptVerificationError(f"recorded artifact changed: {path}")
    return path, data


def _load_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptVerificationError(f"cannot read JSON artifact {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReceiptVerificationError(f"JSON artifact must contain an object: {label}")
    return payload


def _require_fields(value: Any, expected: frozenset[str], *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ReceiptVerificationError(f"{label} field set mismatch")


def _require_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReceiptVerificationError(f"{label} must be an integer")
    return value


def _verify_pilot_contract(*, pilot: Any, seed: int, fold: int, device: Any) -> None:
    if not isinstance(pilot, bool):
        raise ReceiptVerificationError("pilot flag must be boolean")
    if pilot and (seed, fold, device) != (42, 0, "cpu"):
        raise ReceiptVerificationError("pilot evidence must be seed 42, fold 0, CPU")


def _summary_completion_roots(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    roots, evidence = summary.get("accepted_attempt_completion_roots"), summary.get("fold_replay_evidence")
    if not isinstance(roots, list) or not isinstance(evidence, list):
        raise ReceiptVerificationError("summary lacks completion-root evidence")
    normalized = [
        _validate_record(record, label=f"summary completion root[{index}]")
        for index, record in enumerate(roots)
    ]
    evidence_roots: list[dict[str, Any]] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise ReceiptVerificationError(f"fold replay evidence[{index}] is invalid")
        evidence_roots.append(
            _validate_record(item.get("completion"), label=f"fold replay evidence[{index}].completion")
        )
    if normalized != evidence_roots:
        raise ReceiptVerificationError("summary completion roots disagree")
    if summary.get("verification_passed") is True and not normalized:
        raise ReceiptVerificationError("passing summary has no completion roots")
    return normalized


def _verify_completion_tree(record: Mapping[str, Any], *, output_root: Path) -> tuple[int, int]:
    completion_path, completion_bytes = _snapshot_record(record, root=output_root, label="completion root")
    completion = _load_json_bytes(completion_bytes, label=str(completion_path))
    _require_fields(completion, COMPLETION_FIELDS, label="completion")
    if completion.get("schema") != f"{SCHEMA_VERSION}-completion":
        raise ReceiptVerificationError("completion schema mismatch")
    seed = _require_integer(completion.get("seed"), label="completion seed")
    fold = _require_integer(completion.get("fold"), label="completion fold")
    pilot = completion.get("pilot")
    expected_status = "pilot_complete" if pilot is True else "fold_complete"
    if completion.get("run_status") != expected_status:
        raise ReceiptVerificationError("completion status mismatch")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != COMPLETION_ARTIFACT_KEYS:
        raise ReceiptVerificationError("completion artifact key set mismatch")

    attempt = completion_path.parent
    specs = {
        "training": (completion.get("training"), attempt / "training.json"),
        "run_manifest": (completion.get("run_manifest"), attempt / "run-manifest.json"),
        "predictions": (artifacts.get("predictions"), attempt / "outer-test-predictions.npz"),
        "attention_evidence": (artifacts.get("attention_evidence"), attempt / "attention-evidence.npz"),
        "inference": (artifacts.get("inference"), attempt / "inference.json"),
        "metrics": (artifacts.get("metrics"), attempt / "metrics.json"),
    }
    leaves: dict[str, tuple[Path, bytes]] = {}
    for label, (leaf_record, expected) in specs.items():
        leaves[label] = _snapshot_record(
            leaf_record, root=output_root, label=f"completion {label}", expected=expected
        )

    manifest = _load_json_bytes(leaves["run_manifest"][1], label="run manifest")
    _require_fields(manifest, MANIFEST_FIELDS, label="run manifest")
    if manifest.get("schema") != f"{SCHEMA_VERSION}-run-manifest":
        raise ReceiptVerificationError("run-manifest schema mismatch")
    manifest_seed = _require_integer(manifest.get("seed"), label="run-manifest seed")
    manifest_fold = _require_integer(manifest.get("fold"), label="run-manifest fold")
    arguments = manifest.get("arguments")
    if not isinstance(arguments, Mapping) or set(arguments) != MANIFEST_ARGUMENT_FIELDS:
        raise ReceiptVerificationError("run-manifest arguments field set mismatch")
    _verify_pilot_contract(
        pilot=manifest.get("pilot"), seed=manifest_seed, fold=manifest_fold, device=arguments.get("device")
    )
    if (manifest_seed, manifest_fold, manifest.get("pilot")) != (seed, fold, pilot):
        raise ReceiptVerificationError("completion identity disagrees with run manifest")

    training = _load_json_bytes(leaves["training"][1], label="training")
    _require_fields(training, TRAINING_FIELDS, label="training")
    if training.get("schema") != f"{SCHEMA_VERSION}-training":
        raise ReceiptVerificationError("training schema mismatch")
    if (
        _require_integer(training.get("seed"), label="training seed") != seed
        or _require_integer(training.get("fold"), label="training fold") != fold
    ):
        raise ReceiptVerificationError("training identity disagrees with completion")
    _snapshot_record(
        training.get("checkpoint"), root=output_root, label="training checkpoint",
        expected=attempt / "trained-model.pt",
    )

    inference = _load_json_bytes(leaves["inference"][1], label="inference")
    _require_fields(inference, INFERENCE_FIELDS, label="inference")
    if inference.get("schema") != f"{SCHEMA_VERSION}-inference":
        raise ReceiptVerificationError("inference schema mismatch")
    for key in ("predictions", "attention_evidence"):
        if _validate_record(inference.get(key), label=f"inference {key}") != dict(artifacts[key]):
            raise ReceiptVerificationError(f"inference {key} record is not cross-bound")

    metrics = _load_json_bytes(leaves["metrics"][1], label="metrics")
    _require_fields(metrics, METRICS_FIELDS, label="metrics")
    if metrics.get("schema") != f"{SCHEMA_VERSION}-metrics":
        raise ReceiptVerificationError("metrics schema mismatch")

    nested = 1
    for field in ("code", "inputs"):
        records = manifest.get(field)
        if not isinstance(records, list) or not records:
            raise ReceiptVerificationError(f"run manifest {field} must be non-empty")
        for index, nested_record in enumerate(records):
            _snapshot_record(nested_record, root=ROOT, label=f"run manifest {field}[{index}]")
            nested += 1
    return len(leaves), nested


def verify_receipt_without_torch(
    receipt_path: Path, *, output_root: Path | None = None
) -> dict[str, Any]:
    """Replay a detached receipt using only Python's standard library."""

    receipt_path = _absolute_without_resolving(receipt_path)
    _assert_no_link_or_reparse(receipt_path)
    try:
        receipt_bytes = receipt_path.read_bytes()
    except OSError as exc:
        raise ReceiptVerificationError(f"cannot read receipt: {receipt_path}") from exc
    receipt = _load_json_bytes(receipt_bytes, label=str(receipt_path))
    _require_fields(receipt, RECEIPT_FIELDS, label="receipt")
    if receipt.get("schema") != f"{SCHEMA_VERSION}-detached-verification-receipt":
        raise ReceiptVerificationError("receipt schema mismatch")

    recorded_output = receipt.get("source_output_root")
    if not isinstance(recorded_output, str) or not recorded_output:
        raise ReceiptVerificationError("receipt source_output_root is invalid")
    selected_output = Path(output_root) if output_root is not None else Path(recorded_output)
    selected_output = _resolve_path(
        selected_output, label="verification output_root", strict=True
    )
    if not selected_output.is_dir():
        raise ReceiptVerificationError("verification output_root is not a directory")
    receipt_path = _require_external_artifact(
        receipt_path,
        output_root=selected_output,
        label="detached verification receipt",
    )

    summary_record = _validate_record(
        receipt.get("summary"), label="receipt summary"
    )
    summary_recorded_path = Path(str(summary_record["path"]))
    summary_lexical_path = (
        summary_recorded_path
        if summary_recorded_path.is_absolute()
        else receipt_path.parent / summary_recorded_path
    )
    _assert_no_link_or_reparse(summary_lexical_path)
    summary_path, summary_bytes = _snapshot_record(
        summary_record, root=receipt_path.parent, label="receipt summary"
    )
    summary_path = _require_external_artifact(
        summary_path,
        output_root=selected_output,
        label="detached verification summary",
    )
    summary = _load_json_bytes(summary_bytes, label=str(summary_path))
    for field in ("scope", "verification_passed", "scientific_result"):
        if receipt.get(field) != summary.get(field):
            raise ReceiptVerificationError(f"receipt {field} disagrees with summary")
    summary_roots = _summary_completion_roots(summary)
    receipt_roots_value = receipt.get("accepted_attempt_completion_roots")
    if not isinstance(receipt_roots_value, list):
        raise ReceiptVerificationError("receipt completion roots must be a list")
    receipt_roots = [
        _validate_record(record, label=f"receipt completion root[{index}]")
        for index, record in enumerate(receipt_roots_value)
    ]
    if receipt_roots != summary_roots:
        raise ReceiptVerificationError("receipt completion roots disagree with summary")

    direct = 0
    nested = 0
    for record in receipt_roots:
        direct_count, nested_count = _verify_completion_tree(record, output_root=selected_output)
        direct += direct_count
        nested += nested_count

    verifier_records = receipt.get("verifier_code")
    if not isinstance(verifier_records, list) or not verifier_records:
        raise ReceiptVerificationError("receipt verifier_code is invalid")
    for index, record in enumerate(verifier_records):
        _snapshot_record(record, root=ROOT, label=f"receipt verifier_code[{index}]")
    return {
        "schema": f"{SCHEMA_VERSION}-detached-verification-replay",
        "receipt_verified": True,
        "receipt": {
            "path": receipt_path.name, "bytes": len(receipt_bytes),
            "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        },
        "source_output_root": recorded_output,
        "verification_output_root": str(selected_output),
        "scope": summary.get("scope"),
        "verification_passed": summary.get("verification_passed"),
        "scientific_result": summary.get("scientific_result"),
        "n_completion_roots_verified": len(summary_roots),
        "n_direct_leaf_records_verified": direct,
        "n_nested_leaf_records_verified": nested,
        "n_verifier_code_files_verified": len(verifier_records),
        "assurance_scope": (
            "Standard-library-only, read-only checksum/schema replay of an unsigned receipt; "
            "no checkpoint deserialization, independent implementation, signature, or archive "
            "immutability is established."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--receipt-root", type=Path, default=None)
    parser.add_argument("--replay-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--verify-receipt", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.pilot and args.replay_device != "cpu":
        parser.error("pilot verification is CPU-only")
    if args.verify_receipt is not None and (args.pilot or args.receipt_root is not None):
        parser.error("--verify-receipt cannot be combined with --pilot/--receipt-root")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_receipt is not None:
        try:
            report = verify_receipt_without_torch(args.verify_receipt, output_root=args.output_root)
        except (
            ReceiptVerificationError,
            OSError,
            RuntimeError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
        ) as exc:
            print(f"attention receipt verification failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    try:
        faith = _load_faith_module()
    except Exception as exc:
        print(
            "attention verification failed while importing the replay "
            f"implementation: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    try:
        output_root = args.output_root or DEFAULT_OUTPUT_ROOT
        receipt_root = (
            args.receipt_root
            if args.receipt_root is not None
            else faith._default_detached_receipt_root(output_root)
        )
        summary = faith.verify_campaign(
            output_root, pilot=bool(args.pilot), write_summary=False,
            replay_device=args.replay_device, detached_receipt_root=receipt_root,
            verifier_entrypoint=Path(__file__),
        )
    except (
        faith.VerificationError,
        OSError,
        RuntimeError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        ImportError,
    ) as exc:
        print(f"attention verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["verification_passed"] is not True:
        return 2
    if not args.pilot and summary["scientific_result"] is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
