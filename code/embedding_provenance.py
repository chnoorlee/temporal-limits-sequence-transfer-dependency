"""Fail-closed validation for the canonical ESM-2 embedding bundle.

The training and aggregation entry points use this module to bind every
scientific run to the same sequences, model revision, generation program,
environment inventory, and output arrays.  Generation-time modification times
are retained in the manifest for auditability but are not identity fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np


EMBEDDING_SCHEMA = "DepPLM-esm-embedding-v1"
MODEL_REPOSITORY = "facebook/esm2_t6_8M_UR50D"
MODEL_REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
OUTPUT_STEM = "esm2_t6_8M_residue_chunked"
MANIFEST_RELATIVE_PATH = f"data/processed/{OUTPUT_STEM}.manifest.json"
EMBEDDING_RELATIVE_PATH = f"data/processed/{OUTPUT_STEM}.npy"
NORMALIZED_RELATIVE_PATH = f"data/processed/{OUTPUT_STEM}.l2.npy"
NAMES_RELATIVE_PATH = f"data/processed/{OUTPUT_STEM}.names.txt"
FASTA_RELATIVE_PATH = "data/processed/gene_seqs.fa"
GENES_RELATIVE_PATH = "data/processed/genes.txt"
GENERATOR_RELATIVE_PATH = "code/03_embed_proteins.py"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OFFICIAL_SNAPSHOT_FILES: dict[str, tuple[int, str]] = {
    "config.json": (
        775,
        "facb9355e8252149f6f99dff5fd1e32e890c3c655362b261d0c81cd6b5839c85",
    ),
    "model.safetensors": (
        31_384_292,
        "24c5fa474c48f3b754b86efe752d5f189d2bcd88190fa2270fc92b2ef3034189",
    ),
    "special_tokens_map.json": (
        125,
        "3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1",
    ),
    "tokenizer_config.json": (
        95,
        "7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad",
    ),
    "vocab.txt": (
        93,
        "0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03",
    ),
}


class EmbeddingProvenanceError(ValueError):
    """Raised when the canonical embedding bundle cannot be verified."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EmbeddingProvenanceError(message)


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_relative_path(value: Any, *, field: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{field} must be a path string")
    normalized = value.replace("\\", "/")
    parsed = PurePosixPath(normalized)
    _require(
        not parsed.is_absolute()
        and not re.match(r"^[A-Za-z]:", normalized)
        and ".." not in parsed.parts,
        f"{field} must remain inside the workspace",
    )
    return parsed.as_posix()


def _workspace_path(root: Path, relative: str) -> Path:
    root_resolved = root.resolve(strict=True)
    target = (root_resolved / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise EmbeddingProvenanceError(
            f"resolved artifact escapes the workspace: {relative}"
        ) from exc
    return target


def _validate_workspace_file_record(
    record: Any,
    *,
    root: Path,
    expected_relative_path: str,
    field: str,
) -> dict[str, Any]:
    _require(isinstance(record, Mapping), f"{field} must be an object")
    relative = _normalized_relative_path(record.get("path"), field=f"{field}.path")
    _require(
        relative == expected_relative_path,
        f"{field}.path expected {expected_relative_path!r}, found {relative!r}",
    )
    size = record.get("bytes")
    digest = record.get("sha256")
    _require(
        isinstance(size, int) and not isinstance(size, bool) and size > 0,
        f"{field}.bytes must be a positive integer",
    )
    _require(
        isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
        f"{field}.sha256 must be a lowercase SHA-256 digest",
    )
    path = _workspace_path(root, relative)
    _require(path.is_file(), f"{field} is not a regular file: {relative}")
    _require(path.stat().st_size == size, f"{field}.bytes does not match {relative}")
    _require(_sha256_file(path) == digest, f"{field}.sha256 does not match {relative}")
    return {"path": relative, "bytes": size, "sha256": digest}


def _hash_names(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_lines(path: Path, *, field: str) -> list[str]:
    try:
        values = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EmbeddingProvenanceError(f"cannot read {field}: {exc}") from exc
    _require(values and all(values), f"{field} must contain non-empty lines")
    _require(len(values) == len(set(values)), f"{field} contains duplicate identifiers")
    return values


def _load_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    pieces: list[str] = []
    seen: set[str] = set()

    def finish() -> None:
        nonlocal name, pieces
        if name is None:
            return
        sequence = "".join(pieces)
        _require(bool(sequence), f"empty FASTA sequence for {name!r}")
        try:
            sequence.encode("ascii")
        except UnicodeEncodeError as exc:
            raise EmbeddingProvenanceError(
                f"non-ASCII FASTA sequence for {name!r}"
            ) from exc
        records.append((name, sequence))

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EmbeddingProvenanceError(f"cannot read FASTA: {exc}") from exc
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish()
            candidate = line[1:].split("|", 1)[0]
            _require(bool(candidate), f"empty FASTA identifier at line {line_number}")
            _require(candidate not in seen, f"duplicate FASTA identifier: {candidate}")
            seen.add(candidate)
            name = candidate
            pieces = []
        else:
            _require(name is not None, f"sequence precedes FASTA header at line {line_number}")
            _require(
                not any(character.isspace() for character in line),
                f"embedded whitespace in FASTA sequence {name!r}",
            )
            pieces.append(line)
    finish()
    _require(bool(records), "FASTA contains no records")
    return records


def _hash_sequence_records(records: Sequence[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for name, sequence in records:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sequence.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_snapshot_files(value: Any) -> list[dict[str, Any]]:
    _require(isinstance(value, list), "model.snapshot_files must be a list")
    observed: dict[str, tuple[int, str]] = {}
    for index, record in enumerate(value):
        field = f"model.snapshot_files[{index}]"
        _require(isinstance(record, Mapping), f"{field} must be an object")
        path = _normalized_relative_path(record.get("path"), field=f"{field}.path")
        _require("/" not in path, f"{field}.path must be a snapshot basename")
        size = record.get("bytes")
        digest = record.get("sha256")
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size > 0,
            f"{field}.bytes must be positive",
        )
        _require(
            isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
            f"{field}.sha256 is invalid",
        )
        _require(path not in observed, f"duplicate snapshot file record: {path}")
        observed[path] = (size, digest)
    _require(
        observed == _OFFICIAL_SNAPSHOT_FILES,
        "snapshot file sizes/hashes do not match the pinned official ESM-2 revision",
    )
    return [
        {"path": path, "bytes": size, "sha256": digest}
        for path, (size, digest) in sorted(observed.items())
    ]


def _validate_environment(value: Any) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "environment must be an object")
    required_strings = ("python", "platform", "machine", "torch_version")
    for name in required_strings:
        _require(
            isinstance(value.get(name), str) and bool(value[name]),
            f"environment.{name} must be a non-empty string",
        )
    _require(
        value.get("deterministic_algorithms_enabled") is True,
        "embedding generation did not enable deterministic algorithms",
    )
    _require(
        value.get("deterministic_debug_mode") == 2,
        "embedding generation did not use deterministic debug mode 2",
    )
    _require(
        value.get("cudnn_deterministic") is True,
        "embedding generation did not request deterministic cuDNN behavior",
    )
    _require(
        value.get("cudnn_benchmark") is False,
        "embedding generation enabled cuDNN benchmarking",
    )
    installed = value.get("installed_distributions")
    _require(isinstance(installed, list) and bool(installed), "environment inventory is empty")
    inventory: dict[str, str] = {}
    for index, record in enumerate(installed):
        field = f"environment.installed_distributions[{index}]"
        _require(isinstance(record, Mapping), f"{field} must be an object")
        name = record.get("name")
        version = record.get("version")
        _require(isinstance(name, str) and bool(name), f"{field}.name is invalid")
        _require(isinstance(version, str) and bool(version), f"{field}.version is invalid")
        normalized = name.casefold()
        _require(normalized not in inventory, f"duplicate distribution in environment: {name}")
        inventory[normalized] = version
    key_versions = value.get("key_distribution_versions")
    _require(isinstance(key_versions, Mapping), "environment key versions are missing")
    required_packages = ("huggingface_hub", "numpy", "safetensors", "torch", "transformers")
    for package in required_packages:
        version = key_versions.get(package)
        _require(
            isinstance(version, str) and bool(version),
            f"environment.key_distribution_versions.{package} is missing",
        )
        _require(
            inventory.get(package.casefold()) == version,
            f"environment inventory disagrees for {package}",
        )
    inventory_records = [
        {"name": record["name"], "version": record["version"]} for record in installed
    ]
    return {
        "python": value["python"],
        "platform": value["platform"],
        "machine": value["machine"],
        "device": value.get("device"),
        "cuda_available": value.get("cuda_available"),
        "cuda_device_name": value.get("cuda_device_name"),
        "torch_cuda_version": value.get("torch_cuda_version"),
        "cudnn_version": value.get("cudnn_version"),
        "key_distribution_versions": dict(key_versions),
        "installed_distribution_count": len(installed),
        "installed_distributions_sha256": _canonical_sha256(
            {"installed_distributions": inventory_records}
        ),
        "deterministic_algorithms_enabled": True,
        "deterministic_debug_mode": 2,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }


def validate_embedding_bundle(
    root: Path,
    *,
    expected_n_genes: int = 17_768,
    expected_dimension: int = 320,
) -> dict[str, Any]:
    """Validate the canonical bundle and return stable, compact provenance.

    ``expected_n_genes`` and ``expected_dimension`` exist to make the validator
    testable on small synthetic bundles.  Production callers use the fixed
    defaults.
    """
    _require(expected_n_genes > 0, "expected_n_genes must be positive")
    _require(expected_dimension > 0, "expected_dimension must be positive")
    root = root.resolve(strict=True)
    manifest_path = _workspace_path(root, MANIFEST_RELATIVE_PATH)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EmbeddingProvenanceError(f"cannot read embedding manifest: {exc}") from exc
    _require(isinstance(manifest, Mapping), "embedding manifest must be an object")
    _require(manifest.get("schema") == EMBEDDING_SCHEMA, "unexpected embedding schema")
    _require(manifest.get("status") == "complete", "embedding manifest is not complete")

    generator = _validate_workspace_file_record(
        manifest.get("generator"),
        root=root,
        expected_relative_path=GENERATOR_RELATIVE_PATH,
        field="generator",
    )
    input_block = manifest.get("input")
    _require(isinstance(input_block, Mapping), "input provenance is missing")
    fasta_record = _validate_workspace_file_record(
        input_block.get("fasta"),
        root=root,
        expected_relative_path=FASTA_RELATIVE_PATH,
        field="input.fasta",
    )

    model = manifest.get("model")
    _require(isinstance(model, Mapping), "model provenance is missing")
    _require(model.get("repository") == MODEL_REPOSITORY, "unexpected ESM repository")
    _require(model.get("model_revision") == MODEL_REVISION, "unexpected model revision")
    _require(
        model.get("tokenizer_revision") == MODEL_REVISION,
        "tokenizer revision does not match the pinned model revision",
    )
    _require(model.get("model_class") == "EsmModel", "unexpected model class")
    _require(model.get("tokenizer_class") == "EsmTokenizer", "unexpected tokenizer class")
    configuration = model.get("configuration")
    _require(isinstance(configuration, Mapping), "model configuration is missing")
    _require(configuration.get("hidden_size") == expected_dimension, "unexpected ESM hidden size")
    _require(configuration.get("num_hidden_layers") == 6, "unexpected ESM layer count")
    _require(configuration.get("max_position_embeddings") == 1026, "unexpected ESM context size")
    snapshot_files = _validate_snapshot_files(model.get("snapshot_files"))

    generation = manifest.get("generation")
    _require(isinstance(generation, Mapping), "generation protocol is missing")
    expected_generation = {
        "strategy": "chunked",
        "maximum_residues_per_window": 1022,
        "overlap_residues": 128,
        "stride_residues": 894,
        "special_tokens_excluded": True,
        "padding_tokens_excluded": True,
        "truncation_enabled_in_tokenizer": False,
        "output_dtype": "float32",
        "output_shape": [expected_n_genes, expected_dimension],
        "recommended_for_primary_analysis": True,
    }
    for name, expected in expected_generation.items():
        _require(
            generation.get(name) == expected,
            f"generation.{name} expected {expected!r}, found {generation.get(name)!r}",
        )
    _require(
        generation.get("pooling")
        == "inverse-coverage-weighted arithmetic mean of residue hidden states across deterministic overlapping windows",
        "unexpected ESM pooling protocol",
    )
    _require(
        generation.get("long_protein_behavior")
        == "all residues represented; overlap duplicates receive inverse coverage weights so every original residue has total weight one",
        "unexpected long-protein protocol",
    )

    outputs = manifest.get("outputs")
    _require(isinstance(outputs, Mapping), "output provenance is missing")
    output_specs = {
        "embeddings": (EMBEDDING_RELATIVE_PATH, None),
        "normalized_embeddings": (NORMALIZED_RELATIVE_PATH, "row-wise L2 with float32 epsilon floor"),
        "names": (NAMES_RELATIVE_PATH, None),
    }
    output_records: dict[str, dict[str, Any]] = {}
    for name, (relative, normalization) in output_specs.items():
        record = outputs.get(name)
        output_records[name] = _validate_workspace_file_record(
            record,
            root=root,
            expected_relative_path=relative,
            field=f"outputs.{name}",
        )
        _require(isinstance(record, Mapping), f"outputs.{name} must be an object")
        if name == "names":
            _require(record.get("records") == expected_n_genes, "names record count is incorrect")
            _require(record.get("encoding") == "UTF-8", "names encoding is not UTF-8")
        else:
            _require(record.get("dtype") == "float32", f"outputs.{name}.dtype is incorrect")
            _require(
                record.get("shape") == [expected_n_genes, expected_dimension],
                f"outputs.{name}.shape is incorrect",
            )
            _require(record.get("numpy_allow_pickle") is False, f"outputs.{name} permits pickle")
        if normalization is not None:
            _require(record.get("normalization") == normalization, "normalization record is incorrect")

    genes = _load_lines(_workspace_path(root, GENES_RELATIVE_PATH), field=GENES_RELATIVE_PATH)
    names = _load_lines(_workspace_path(root, NAMES_RELATIVE_PATH), field=NAMES_RELATIVE_PATH)
    _require(len(genes) == expected_n_genes, "genes.txt has an unexpected record count")
    _require(names == genes, "embedding names/order do not exactly match genes.txt")
    _require(
        input_block.get("sequence_names_sha256") == _hash_names(names),
        "sequence_names_sha256 does not match the names artifact",
    )

    fasta = _load_fasta(_workspace_path(root, FASTA_RELATIVE_PATH))
    fasta_names = [name for name, _ in fasta]
    _require(fasta_names == genes, "FASTA identifiers/order do not exactly match genes.txt")
    _require(
        input_block.get("sequence_order_and_content_sha256")
        == _hash_sequence_records(fasta),
        "sequence order/content digest does not match the FASTA",
    )
    lengths = [len(sequence) for _, sequence in fasta]
    long_names = [name for name, sequence in fasta if len(sequence) > 1022]
    statistics_record = input_block.get("statistics")
    _require(isinstance(statistics_record, Mapping), "input sequence statistics are missing")
    expected_statistics: dict[str, Any] = {
        "n_sequences": expected_n_genes,
        "n_sequences_longer_than_window": len(long_names),
        "minimum_length_residues": min(lengths),
        "maximum_length_residues": max(lengths),
        "median_length_residues": float(statistics.median(lengths)),
        "long_sequence_names_sha256": _hash_names(long_names),
    }
    for name, expected in expected_statistics.items():
        _require(statistics_record.get(name) == expected, f"input.statistics.{name} is incorrect")
    fraction = statistics_record.get("fraction_sequences_longer_than_window")
    _require(
        isinstance(fraction, (int, float))
        and not isinstance(fraction, bool)
        and math.isclose(float(fraction), len(long_names) / expected_n_genes, rel_tol=0.0, abs_tol=1e-15),
        "input.statistics.fraction_sequences_longer_than_window is incorrect",
    )

    embedding = np.load(_workspace_path(root, EMBEDDING_RELATIVE_PATH), mmap_mode="r", allow_pickle=False)
    normalized = np.load(_workspace_path(root, NORMALIZED_RELATIVE_PATH), mmap_mode="r", allow_pickle=False)
    expected_shape = (expected_n_genes, expected_dimension)
    _require(embedding.shape == expected_shape, "embedding array shape is incorrect")
    _require(normalized.shape == expected_shape, "normalized embedding array shape is incorrect")
    _require(embedding.dtype == np.dtype("float32"), "embedding array dtype is not float32")
    _require(normalized.dtype == np.dtype("float32"), "normalized embedding dtype is not float32")
    for start in range(0, expected_n_genes, 2048):
        end = min(start + 2048, expected_n_genes)
        raw_block = np.asarray(embedding[start:end], dtype=np.float32)
        normalized_block = np.asarray(normalized[start:end], dtype=np.float32)
        _require(np.isfinite(raw_block).all(), "embedding array contains non-finite values")
        _require(np.isfinite(normalized_block).all(), "normalized embedding contains non-finite values")
        raw_norms = np.linalg.norm(raw_block, axis=1, keepdims=True)
        _require(np.all(raw_norms > np.finfo(np.float32).eps), "embedding contains a zero-norm row")
        expected_normalized = raw_block / np.maximum(raw_norms, np.finfo(np.float32).eps)
        _require(
            np.allclose(normalized_block, expected_normalized, rtol=2e-6, atol=2e-6),
            "normalized embeddings are not the row-wise L2 transform of raw embeddings",
        )
        normalized_norms = np.linalg.norm(normalized_block, axis=1)
        _require(
            np.allclose(normalized_norms, 1.0, rtol=2e-6, atol=2e-6),
            "normalized embedding rows do not have unit L2 norm",
        )

    environment = _validate_environment(manifest.get("environment"))
    manifest_record = {
        "path": MANIFEST_RELATIVE_PATH,
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha256_file(manifest_path),
    }
    provenance: dict[str, Any] = {
        "schema": "DepPLM-verified-embedding-provenance-v1",
        "verification_status": "verified",
        "manifest": manifest_record,
        "model": {
            "repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "model_class": "EsmModel",
            "tokenizer_class": "EsmTokenizer",
            "hidden_size": expected_dimension,
            "num_hidden_layers": 6,
            "snapshot_files": snapshot_files,
        },
        "generation": {
            key: generation[key]
            for key in (
                "strategy",
                "maximum_residues_per_window",
                "overlap_residues",
                "stride_residues",
                "pooling",
                "long_protein_behavior",
                "special_tokens_excluded",
                "padding_tokens_excluded",
                "truncation_enabled_in_tokenizer",
                "output_dtype",
                "output_shape",
                "recommended_for_primary_analysis",
            )
        },
        "generator": generator,
        "input": {
            "fasta": fasta_record,
            "sequence_names_sha256": input_block["sequence_names_sha256"],
            "sequence_order_and_content_sha256": input_block[
                "sequence_order_and_content_sha256"
            ],
            "statistics": dict(statistics_record),
        },
        "outputs": output_records,
        "environment": environment,
    }
    provenance["provenance_sha256"] = _canonical_sha256(provenance)
    return provenance

