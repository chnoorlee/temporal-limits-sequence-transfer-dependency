"""Generate provenance-bound ESM-2 embeddings.

The default artifact uses deterministic overlapping windows for proteins longer
than the ESM-2 residue limit and pools residue tokens only. The historical
``esm2_t6_8M.npy`` artifact is never overwritten by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODEL_ID = "facebook/esm2_t6_8M_UR50D"
MODEL_REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
MAX_RESIDUES = 1022
DEFAULT_OVERLAP = 128
LEGACY_OUTPUT_STEM = "esm2_t6_8M"
DEFAULT_OUTPUT_STEMS = {
    "chunked": "esm2_t6_8M_residue_chunked",
    "legacy-first-1022": "esm2_t6_8M_residue_legacy_first1022",
}
SNAPSHOT_ALLOW_PATTERNS = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
)


@dataclass(frozen=True)
class SequenceRecord:
    name: str
    sequence: str


@dataclass(frozen=True)
class ChunkRecord:
    sequence_index: int
    start: int
    end: int
    fragment: str
    residue_weights: np.ndarray


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    logical_path = Path(os.path.abspath(path))
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    if relative_to is None:
        display = str(logical_path)
    else:
        try:
            display = str(
                logical_path.relative_to(Path(os.path.abspath(relative_to)))
            )
        except ValueError:
            display = str(logical_path)
    return {
        "path": display,
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(resolved),
    }


def _records_stable(
    initial: Sequence[dict[str, Any]], paths: Sequence[Path], *, label: str
) -> None:
    current = [_file_record(path, relative_to=ROOT) for path in paths]
    if list(initial) != current:
        raise RuntimeError(f"{label} changed during embedding generation")


def _hash_sequence_records(records: Sequence[SequenceRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.sequence.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_names(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_sequences(path: Path) -> list[SequenceRecord]:
    """Parse the processed FASTA without silently changing residues or order."""
    records: list[SequenceRecord] = []
    name: str | None = None
    pieces: list[str] = []
    seen: set[str] = set()

    def finish() -> None:
        nonlocal name, pieces
        if name is None:
            return
        sequence = "".join(pieces)
        if not sequence:
            raise ValueError(f"empty sequence for FASTA record {name!r}")
        try:
            sequence.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"non-ASCII residue data for FASTA record {name!r}") from exc
        records.append(SequenceRecord(name=name, sequence=sequence))

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                header = line[1:]
                candidate = header.split("|", 1)[0]
                if not candidate:
                    raise ValueError(f"empty FASTA identifier at line {line_number}")
                if candidate in seen:
                    raise ValueError(f"duplicate FASTA identifier: {candidate}")
                seen.add(candidate)
                name = candidate
                pieces = []
            else:
                if name is None:
                    raise ValueError(
                        f"sequence data precedes the first FASTA header at line {line_number}"
                    )
                if any(character.isspace() for character in line):
                    raise ValueError(
                        f"embedded whitespace in sequence {name!r} at line {line_number}"
                    )
                pieces.append(line)
    finish()
    if not records:
        raise ValueError(f"no sequences found in {path}")
    return records


def _chunk_windows(length: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if length < 1:
        raise ValueError("sequence length must be positive")
    if not 1 <= chunk_size <= MAX_RESIDUES:
        raise ValueError(f"chunk_size must be in [1, {MAX_RESIDUES}]")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if length <= chunk_size:
        return [(0, length)]
    stride = chunk_size - overlap
    starts = list(range(0, length - chunk_size + 1, stride))
    final_start = length - chunk_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, start + chunk_size) for start in starts]


def _sequence_chunks(
    record: SequenceRecord,
    sequence_index: int,
    *,
    strategy: str,
    chunk_size: int,
    overlap: int,
) -> list[ChunkRecord]:
    length = len(record.sequence)
    if strategy == "legacy-first-1022":
        if chunk_size != MAX_RESIDUES or overlap != 0:
            raise ValueError(
                "legacy-first-1022 requires chunk_size=1022 and overlap=0"
            )
        end = min(length, chunk_size)
        return [
            ChunkRecord(
                sequence_index=sequence_index,
                start=0,
                end=end,
                fragment=record.sequence[:end],
                residue_weights=np.ones(end, dtype=np.float64),
            )
        ]
    if strategy != "chunked":
        raise ValueError(f"unsupported long-sequence strategy: {strategy}")
    windows = _chunk_windows(length, chunk_size, overlap)
    coverage = np.zeros(length, dtype=np.int32)
    for start, end in windows:
        coverage[start:end] += 1
    if np.any(coverage < 1):
        raise RuntimeError("chunk plan does not cover every residue")
    return [
        ChunkRecord(
            sequence_index=sequence_index,
            start=start,
            end=end,
            fragment=record.sequence[start:end],
            residue_weights=(1.0 / coverage[start:end]).astype(np.float64),
        )
        for start, end in windows
    ]


def iter_chunks(
    records: Sequence[SequenceRecord],
    *,
    strategy: str,
    chunk_size: int,
    overlap: int,
) -> Iterator[ChunkRecord]:
    for sequence_index, record in enumerate(records):
        yield from _sequence_chunks(
            record,
            sequence_index,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )


def _batched(values: Iterable[ChunkRecord], batch_size: int) -> Iterator[list[ChunkRecord]]:
    batch: list[ChunkRecord] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _total_chunks(
    records: Sequence[SequenceRecord], *, strategy: str, chunk_size: int, overlap: int
) -> int:
    if strategy == "legacy-first-1022":
        return len(records)
    return sum(len(_chunk_windows(len(record.sequence), chunk_size, overlap)) for record in records)


def embed_sequences(
    tokenizer: Any,
    model: Any,
    records: Sequence[SequenceRecord],
    *,
    batch_size: int,
    device: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
    progress: bool = True,
) -> np.ndarray:
    """Embed sequences while excluding every tokenizer-marked special token.

    For overlapping windows, each occurrence of residue ``i`` receives weight
    ``1 / coverage(i)``. Every original residue therefore contributes total
    weight one to the final whole-protein mean.
    """
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not records:
        raise ValueError("records must not be empty")
    if strategy == "legacy-first-1022" and (
        chunk_size != MAX_RESIDUES or overlap != 0
    ):
        raise ValueError("legacy-first-1022 requires chunk_size=1022 and overlap=0")
    total_chunks = _total_chunks(
        records, strategy=strategy, chunk_size=chunk_size, overlap=overlap
    )
    accumulators: np.ndarray | None = None
    denominators = np.zeros(len(records), dtype=np.float64)
    completed_chunks = 0
    started = time.time()

    model.eval()
    with torch.inference_mode():
        for chunk_batch in _batched(
            iter_chunks(
                records,
                strategy=strategy,
                chunk_size=chunk_size,
                overlap=overlap,
            ),
            batch_size,
        ):
            encoded = tokenizer(
                [chunk.fragment for chunk in chunk_batch],
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            if "attention_mask" not in encoded or "special_tokens_mask" not in encoded:
                raise RuntimeError(
                    "tokenizer must return attention_mask and special_tokens_mask"
                )
            special_tokens_mask = encoded.pop("special_tokens_mask")
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            special_tokens_mask = special_tokens_mask.to(device)
            output = model(**model_inputs).last_hidden_state
            residue_mask = (
                model_inputs["attention_mask"].bool() & ~special_tokens_mask.bool()
            )
            if output.ndim != 3:
                raise RuntimeError("ESM hidden state must have shape (batch, tokens, hidden)")
            if accumulators is None:
                accumulators = np.zeros(
                    (len(records), int(output.shape[-1])), dtype=np.float64
                )
            for row, chunk in enumerate(chunk_batch):
                positions = torch.nonzero(residue_mask[row], as_tuple=False).flatten()
                if int(positions.numel()) != len(chunk.fragment):
                    raise RuntimeError(
                        "tokenizer residue-token count does not match input residues for "
                        f"{records[chunk.sequence_index].name!r}: "
                        f"{int(positions.numel())} tokens versus {len(chunk.fragment)} residues"
                    )
                hidden = output[row, positions].detach().float().cpu().numpy()
                weighted = np.sum(hidden * chunk.residue_weights[:, None], axis=0)
                accumulators[chunk.sequence_index] += weighted
                denominators[chunk.sequence_index] += float(
                    np.sum(chunk.residue_weights, dtype=np.float64)
                )
            completed_chunks += len(chunk_batch)
            if progress and (completed_chunks == total_chunks or completed_chunks % 50 == 0):
                elapsed = time.time() - started
                rate = elapsed / max(completed_chunks, 1)
                remaining = rate * (total_chunks - completed_chunks)
                print(
                    f"[esm] chunks {completed_chunks}/{total_chunks} elapsed={elapsed:.0f}s "
                    f"remaining_estimate={remaining:.0f}s",
                    flush=True,
                )

    if accumulators is None or np.any(denominators <= 0):
        raise RuntimeError("embedding generation produced no residue representations")
    vectors = accumulators / denominators[:, None]
    if not np.isfinite(vectors).all():
        raise RuntimeError("embedding generation produced non-finite values")
    return vectors.astype(np.float32)


def _snapshot_records(snapshot_path: Path) -> list[dict[str, Any]]:
    files = sorted(path for path in snapshot_path.rglob("*") if path.is_file())
    names = {path.relative_to(snapshot_path).as_posix() for path in files}
    if "config.json" not in names or "model.safetensors" not in names:
        raise RuntimeError("pinned snapshot lacks config.json or model.safetensors")
    tokenizer_files = {"tokenizer_config.json", "special_tokens_map.json", "vocab.txt"}
    if not tokenizer_files.issubset(names):
        missing = sorted(tokenizer_files - names)
        raise RuntimeError(f"pinned snapshot lacks tokenizer files: {missing}")
    return [_file_record(path, relative_to=snapshot_path) for path in files]


def resolve_snapshot(*, local_files_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=list(SNAPSHOT_ALLOW_PATTERNS),
        local_files_only=local_files_only,
    )
    snapshot = Path(path).resolve(strict=True)
    _snapshot_records(snapshot)
    return snapshot


def load_pinned_model(snapshot_path: Path, *, device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    checkpoint_model = AutoModelForMaskedLM.from_pretrained(
        snapshot_path, local_files_only=True, use_safetensors=True
    )
    model = checkpoint_model.base_model.to(device)
    return tokenizer, model.eval()


def _distribution_inventory() -> list[dict[str, str]]:
    packages: dict[tuple[str, str], dict[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or "UNKNOWN"
        version = distribution.version
        packages[(name.casefold(), version)] = {"name": name, "version": version}
    return [packages[key] for key in sorted(packages)]


def _key_distribution_versions() -> dict[str, str | None]:
    distributions = {
        "numpy": "numpy",
        "torch": "torch",
        "transformers": "transformers",
        "huggingface_hub": "huggingface-hub",
        "safetensors": "safetensors",
    }
    versions: dict[str, str | None] = {}
    for manifest_name, distribution_name in distributions.items():
        try:
            versions[manifest_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            versions[manifest_name] = None
    return versions


def _nvidia_smi_record() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "query": command[1],
        "format": command[2],
        "rows": [line.strip() for line in completed.stdout.splitlines() if line.strip()],
    }


def _environment_record(torch: Any, *, device: str) -> dict[str, Any]:
    relevant_environment = {
        name: os.environ.get(name)
        for name in (
            "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_VISIBLE_DEVICES",
            "PYTORCH_CUDA_ALLOC_CONF",
        )
    }
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device": device,
        "torch_version": torch.__version__,
        "torch_build_config": torch.__config__.show(),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cuda_device_capability": list(torch.cuda.get_device_capability(0))
        if device == "cuda"
        else None,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "numerical_environment": relevant_environment,
        "nvidia_smi": _nvidia_smi_record(),
        "key_distribution_versions": _key_distribution_versions(),
        "installed_distributions": _distribution_inventory(),
    }


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    )
    path = Path(handle.name)
    handle.close()
    return path


def _install_no_overwrite(temp_path: Path, target: Path) -> None:
    """Publish a complete temporary file while refusing to replace any target."""
    try:
        os.link(temp_path, target)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def _write_npy_temp(target: Path, array: np.ndarray) -> Path:
    temp_path = _temporary_path(target)
    try:
        with temp_path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _write_text_temp(target: Path, text: str) -> Path:
    temp_path = _temporary_path(target)
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        return temp_path
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _write_json_no_overwrite(target: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temp_path = _write_text_temp(target, text)
    _install_no_overwrite(temp_path, target)


def _device_from_argument(value: str, torch: Any) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return value


def _output_paths(output_dir: Path, output_stem: str) -> dict[str, Path]:
    if not output_stem or Path(output_stem).name != output_stem:
        raise ValueError("--output-stem must be a simple filename stem")
    if Path(output_stem).suffix:
        raise ValueError("--output-stem must not include a filename suffix")
    if output_stem == LEGACY_OUTPUT_STEM:
        raise ValueError(
            f"refusing reserved legacy output stem {LEGACY_OUTPUT_STEM!r}; "
            "the unverified historical artifact must be preserved"
        )
    return {
        "embeddings": output_dir / f"{output_stem}.npy",
        "normalized_embeddings": output_dir / f"{output_stem}.l2.npy",
        "names": output_dir / f"{output_stem}.names.txt",
        "manifest": output_dir / f"{output_stem}.manifest.json",
    }


def _sequence_statistics(
    records: Sequence[SequenceRecord], *, chunk_size: int
) -> dict[str, Any]:
    lengths = np.asarray([len(record.sequence) for record in records], dtype=np.int64)
    long_names = [
        record.name for record, length in zip(records, lengths) if int(length) > chunk_size
    ]
    return {
        "n_sequences": int(lengths.size),
        "minimum_length_residues": int(lengths.min()),
        "median_length_residues": float(np.median(lengths)),
        "maximum_length_residues": int(lengths.max()),
        "n_sequences_longer_than_window": len(long_names),
        "fraction_sequences_longer_than_window": float(len(long_names) / lengths.size),
        "long_sequence_names_sha256": _hash_names(long_names),
    }


def _configure_determinism(torch: Any) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_generation(args: argparse.Namespace, *, argv: Sequence[str] | None = None) -> Path:
    import torch

    started = time.time()
    _configure_determinism(torch)
    device = _device_from_argument(args.device, torch)
    overlap = args.overlap
    if overlap is None:
        overlap = DEFAULT_OVERLAP if args.strategy == "chunked" else 0
    _chunk_windows(MAX_RESIDUES + 1, args.chunk_size, overlap)
    if args.strategy == "legacy-first-1022" and (
        args.chunk_size != MAX_RESIDUES or overlap != 0
    ):
        raise ValueError(
            "legacy-first-1022 requires --chunk-size 1022 and --overlap 0"
        )

    fasta_path = args.fasta.resolve(strict=True)
    script_path = Path(__file__).resolve(strict=True)
    watched_paths = [fasta_path, script_path]
    watched_initial = [_file_record(path, relative_to=ROOT) for path in watched_paths]
    records = load_sequences(fasta_path)

    output_stem = args.output_stem or DEFAULT_OUTPUT_STEMS[args.strategy]
    paths = _output_paths(args.output_dir.resolve(), output_stem)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")

    snapshot_path = resolve_snapshot(local_files_only=args.local_files_only)
    snapshot_initial = _snapshot_records(snapshot_path)
    tokenizer, model = load_pinned_model(snapshot_path, device=device)
    vectors = embed_sequences(
        tokenizer,
        model,
        records,
        batch_size=args.batch_size,
        device=device,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=overlap,
    )
    normalized = vectors / np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True), np.finfo(np.float32).eps
    )
    normalized = normalized.astype(np.float32)

    _records_stable(watched_initial, watched_paths, label="FASTA input or generator code")
    if snapshot_initial != _snapshot_records(snapshot_path):
        raise RuntimeError("pinned model snapshot changed during embedding generation")

    temporary: dict[str, Path] = {}
    published: list[Path] = []
    manifest_published = False
    try:
        temporary["embeddings"] = _write_npy_temp(paths["embeddings"], vectors)
        temporary["normalized_embeddings"] = _write_npy_temp(
            paths["normalized_embeddings"], normalized
        )
        temporary["names"] = _write_text_temp(
            paths["names"], "\n".join(record.name for record in records) + "\n"
        )
        _records_stable(
            watched_initial, watched_paths, label="FASTA input or generator code"
        )
        if snapshot_initial != _snapshot_records(snapshot_path):
            raise RuntimeError("pinned model snapshot changed during output serialization")

        for key in ("embeddings", "normalized_embeddings", "names"):
            _install_no_overwrite(temporary.pop(key), paths[key])
            published.append(paths[key])

        _records_stable(
            watched_initial, watched_paths, label="FASTA input or generator code"
        )
        if snapshot_initial != _snapshot_records(snapshot_path):
            raise RuntimeError("pinned model snapshot changed before manifest publication")

        model_config = model.config.to_dict() if hasattr(model, "config") else None
        manifest = {
            "schema": "DepPLM-esm-embedding-v1",
            "status": "complete",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": time.time() - started,
            "argv": list(sys.argv if argv is None else argv),
            "model": {
                "repository": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": MODEL_REVISION,
                "snapshot_path": str(snapshot_path),
                "snapshot_files": snapshot_initial,
                "model_class": type(model).__name__,
                "tokenizer_class": type(tokenizer).__name__,
                "configuration": model_config,
            },
            "input": {
                "fasta": watched_initial[0],
                "sequence_order_and_content_sha256": _hash_sequence_records(records),
                "sequence_names_sha256": _hash_names([record.name for record in records]),
                "statistics": _sequence_statistics(records, chunk_size=args.chunk_size),
            },
            "generation": {
                "strategy": args.strategy,
                "recommended_for_primary_analysis": args.strategy == "chunked",
                "maximum_residues_per_window": int(args.chunk_size),
                "overlap_residues": int(overlap),
                "stride_residues": int(args.chunk_size - overlap),
                "pooling": (
                    "inverse-coverage-weighted arithmetic mean of residue hidden "
                    "states across deterministic overlapping windows"
                    if args.strategy == "chunked"
                    else "arithmetic mean of residue hidden states in first 1022 residues"
                ),
                "special_tokens_excluded": True,
                "padding_tokens_excluded": True,
                "truncation_enabled_in_tokenizer": False,
                "long_protein_behavior": (
                    "all residues represented; overlap duplicates receive inverse "
                    "coverage weights so every original residue has total weight one"
                    if args.strategy == "chunked"
                    else "residues after position 1022 are omitted, matching the legacy policy"
                ),
                "batch_size_chunks": int(args.batch_size),
                "output_shape": list(vectors.shape),
                "output_dtype": str(vectors.dtype),
            },
            "outputs": {
                "embeddings": {
                    **_file_record(paths["embeddings"], relative_to=ROOT),
                    "shape": list(vectors.shape),
                    "dtype": str(vectors.dtype),
                    "numpy_allow_pickle": False,
                },
                "normalized_embeddings": {
                    **_file_record(
                        paths["normalized_embeddings"], relative_to=ROOT
                    ),
                    "shape": list(normalized.shape),
                    "dtype": str(normalized.dtype),
                    "normalization": "row-wise L2 with float32 epsilon floor",
                    "numpy_allow_pickle": False,
                },
                "names": {
                    **_file_record(paths["names"], relative_to=ROOT),
                    "records": len(records),
                    "encoding": "UTF-8",
                    "line_ending": "LF",
                },
            },
            "generator": watched_initial[1],
            "environment": _environment_record(torch, device=device),
        }
        _write_json_no_overwrite(paths["manifest"], manifest)
        manifest_published = True
    except Exception:
        for temp_path in temporary.values():
            temp_path.unlink(missing_ok=True)
        if not manifest_published:
            for path in published:
                path.unlink(missing_ok=True)
        raise

    print(f"[esm] complete shape={vectors.shape} manifest={paths['manifest']}", flush=True)
    return paths["manifest"]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, default=PROCESSED / "gene_seqs.fa")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    parser.add_argument(
        "--output-stem",
        help="simple stem; defaults to a strategy-specific name that preserves legacy files",
    )
    parser.add_argument("--strategy", choices=sorted(DEFAULT_OUTPUT_STEMS), default="chunked")
    parser.add_argument("--chunk-size", type=_positive_int, default=MAX_RESIDUES)
    parser.add_argument(
        "--overlap",
        type=_nonnegative_int,
        default=None,
        help=f"default: {DEFAULT_OVERLAP} for chunked, 0 for legacy-first-1022",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require the pinned Hugging Face snapshot to be present in the local cache",
    )
    args = parser.parse_args(argv)
    if args.chunk_size > MAX_RESIDUES:
        parser.error(f"--chunk-size must not exceed {MAX_RESIDUES}")
    overlap = args.overlap
    if overlap is not None and overlap >= args.chunk_size:
        parser.error("--overlap must be smaller than --chunk-size")
    if args.strategy == "legacy-first-1022" and args.chunk_size != MAX_RESIDUES:
        parser.error("--strategy legacy-first-1022 requires --chunk-size 1022")
    if args.strategy == "legacy-first-1022" and overlap not in (None, 0):
        parser.error("--strategy legacy-first-1022 requires --overlap 0")
    if args.output_stem == LEGACY_OUTPUT_STEM:
        parser.error(f"--output-stem {LEGACY_OUTPUT_STEM!r} is reserved")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_generation(args, argv=argv)


if __name__ == "__main__":
    main()
