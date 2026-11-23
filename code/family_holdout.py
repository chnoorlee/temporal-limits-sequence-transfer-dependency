#!/usr/bin/env python3
"""Build versioned, leakage-resistant protein family holdout manifests.

The search is deliberately run once at the least stringent registered identity
threshold. All stricter sensitivity analyses are derived from the same frozen,
canonical pair table.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "depplm-family-holdout-v1"
MMSEQS_RELEASE = "18-8cc5c"
MMSEQS_COMMIT = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
REGISTERED_IDENTITIES = (20.0, 30.0, 40.0, 50.0)
MAIN_IDENTITY = 30.0
MIN_BIDIRECTIONAL_COVERAGE = 0.80
SEARCH_EVALUE = 1e-5
SEARCH_SENSITIVITY = 7.5
SEARCH_MAX_SEQS = 20_000
FOLD_COUNT = 5
FOLD_SEED = 42
MMSEQS_FIELDS = (
    "query,target,pident,alnlen,qstart,qend,qlen,tstart,tend,tlen,"
    "qcov,tcov,evalue,bits"
)


class FamilyHoldoutError(RuntimeError):
    """Raised when a family-holdout artifact cannot be trusted."""


@dataclass(frozen=True)
class SequenceRecord:
    gene: str
    accession: str
    sequence: str
    sequence_sha256: str


@dataclass(frozen=True)
class PairEvidence:
    gene_a: str
    gene_b: str
    pident_percent: float
    coverage_a: float
    coverage_b: float
    alignment_length: int
    evalue: float
    bits: float

    def qualifies(self, identity_percent: float) -> bool:
        return (
            self.pident_percent + 1e-9 >= identity_percent
            and self.coverage_a + 1e-9 >= MIN_BIDIRECTIONAL_COVERAGE
            and self.coverage_b + 1e-9 >= MIN_BIDIRECTIONAL_COVERAGE
        )


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in self.parent}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while value != parent:
            next_value = self.parent[value]
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if (self.size[root_left], root_left) < (self.size[root_right], root_right):
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FamilyHoldoutError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write(path, _canonical_json_bytes(value))


def parse_fasta(path: Path) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def finish_record() -> None:
        nonlocal header, sequence_parts
        if header is None:
            return
        token = header.split()[0]
        if "|" not in token:
            raise FamilyHoldoutError(f"FASTA header lacks gene|accession: {header!r}")
        gene, accession = token.split("|", 1)
        sequence = "".join(sequence_parts).upper()
        if not gene or not accession or not sequence:
            raise FamilyHoldoutError(f"invalid FASTA record: {header!r}")
        if any(character.isspace() for character in sequence):
            raise FamilyHoldoutError(f"sequence contains whitespace: {gene}")
        records.append(
            SequenceRecord(
                gene=gene,
                accession=accession,
                sequence=sequence,
                sequence_sha256=hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            )
        )
        header = None
        sequence_parts = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish_record()
                header = line[1:]
            elif header is None:
                raise FamilyHoldoutError(
                    f"sequence before first FASTA header at line {line_number}"
                )
            else:
                sequence_parts.append(line)
    finish_record()
    if not records:
        raise FamilyHoldoutError(f"FASTA contains no records: {path}")
    genes = [record.gene for record in records]
    if len(genes) != len(set(genes)):
        duplicates = sorted(gene for gene in set(genes) if genes.count(gene) > 1)
        raise FamilyHoldoutError(f"duplicate FASTA gene identifiers: {duplicates[:10]}")
    return records


def read_expected_genes(path: Path) -> list[str]:
    genes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    genes = [gene for gene in genes if gene]
    if not genes or len(genes) != len(set(genes)):
        raise FamilyHoldoutError("expected-gene file is empty or contains duplicates")
    return genes


def validate_gene_universe(
    records: Sequence[SequenceRecord], expected_genes: Sequence[str]
) -> None:
    observed = [record.gene for record in records]
    if observed != list(expected_genes):
        missing = sorted(set(expected_genes) - set(observed))
        extra = sorted(set(observed) - set(expected_genes))
        order_only = not missing and not extra
        raise FamilyHoldoutError(
            "FASTA and expected-gene file do not match exactly: "
            f"missing={missing[:10]}, extra={extra[:10]}, order_only={order_only}"
        )


def _parse_mmseqs_identifier(value: str) -> str:
    return value.split()[0].split("|", 1)[0]


def _canonicalize_pair(
    query: str,
    target: str,
    pident: float,
    qcov: float,
    tcov: float,
    alnlen: int,
    evalue: float,
    bits: float,
) -> PairEvidence:
    if query <= target:
        return PairEvidence(query, target, pident, qcov, tcov, alnlen, evalue, bits)
    return PairEvidence(target, query, pident, tcov, qcov, alnlen, evalue, bits)


def _evidence_rank(evidence: PairEvidence) -> tuple[float, ...]:
    return (
        evidence.pident_percent,
        min(evidence.coverage_a, evidence.coverage_b),
        evidence.bits,
        float(evidence.alignment_length),
        -evidence.evalue,
        evidence.coverage_a,
        evidence.coverage_b,
    )


def parse_mmseqs_hits(
    path: Path, gene_universe: Iterable[str]
) -> tuple[list[PairEvidence], dict[str, int]]:
    allowed = set(gene_universe)
    best_by_pair: dict[tuple[str, str], PairEvidence] = {}
    row_count = 0
    self_count = 0
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_count, row in enumerate(reader, start=1):
            if len(row) != 14:
                raise FamilyHoldoutError(
                    f"MMseqs row {row_count} has {len(row)} columns; expected 14"
                )
            query = _parse_mmseqs_identifier(row[0])
            target = _parse_mmseqs_identifier(row[1])
            if query not in allowed or target not in allowed:
                raise FamilyHoldoutError(
                    f"MMseqs row {row_count} has an unknown gene: {query}, {target}"
                )
            try:
                pident = float(row[2])
                alnlen = int(row[3])
                qcov = float(row[10])
                tcov = float(row[11])
                evalue = float(row[12])
                bits = float(row[13])
            except ValueError as error:
                raise FamilyHoldoutError(
                    f"MMseqs row {row_count} contains a non-numeric value"
                ) from error
            if not (0.0 <= pident <= 100.0):
                raise FamilyHoldoutError(f"invalid pident at MMseqs row {row_count}")
            if not (0.0 <= qcov <= 1.0 and 0.0 <= tcov <= 1.0):
                raise FamilyHoldoutError(f"invalid coverage at MMseqs row {row_count}")
            if alnlen <= 0 or evalue < 0.0:
                raise FamilyHoldoutError(f"invalid alignment at MMseqs row {row_count}")
            if query == target:
                self_count += 1
                continue
            evidence = _canonicalize_pair(
                query, target, pident, qcov, tcov, alnlen, evalue, bits
            )
            pair = (evidence.gene_a, evidence.gene_b)
            previous = best_by_pair.get(pair)
            if previous is None or _evidence_rank(evidence) > _evidence_rank(previous):
                best_by_pair[pair] = evidence
    if row_count == 0:
        raise FamilyHoldoutError("MMseqs output contains no rows")
    pairs = [best_by_pair[key] for key in sorted(best_by_pair)]
    return pairs, {
        "raw_alignment_rows": row_count,
        "self_alignment_rows": self_count,
        "canonical_nonself_pairs": len(pairs),
    }


def exact_duplicate_groups(
    records: Sequence[SequenceRecord],
) -> list[tuple[str, ...]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_hash[record.sequence_sha256].append(record.gene)
    return sorted(tuple(sorted(genes)) for genes in by_hash.values() if len(genes) > 1)


def validate_exact_duplicate_evidence(
    records: Sequence[SequenceRecord], pairs: Sequence[PairEvidence]
) -> None:
    evidence_by_pair = {(pair.gene_a, pair.gene_b): pair for pair in pairs}
    missing: list[tuple[str, str]] = []
    invalid: list[tuple[str, str]] = []
    for group in exact_duplicate_groups(records):
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                key = tuple(sorted((left, right)))
                evidence = evidence_by_pair.get(key)
                if evidence is None:
                    missing.append(key)
                elif not evidence.qualifies(100.0 - 1e-6):
                    invalid.append(key)
    if missing or invalid:
        raise FamilyHoldoutError(
            "exact duplicate sequences lack valid full-length search evidence: "
            f"missing={missing}, invalid={invalid}"
        )


def component_identifier(genes: Sequence[str]) -> str:
    payload = "\0".join(sorted(genes)).encode("utf-8")
    return f"fhc-{hashlib.sha256(payload).hexdigest()[:20]}"


def build_components(
    genes: Iterable[str],
    pairs: Sequence[PairEvidence],
    identity_percent: float,
) -> list[tuple[str, tuple[str, ...]]]:
    ordered_genes = sorted(set(genes))
    union_find = UnionFind(ordered_genes)
    for pair in pairs:
        if pair.qualifies(identity_percent):
            union_find.union(pair.gene_a, pair.gene_b)
    grouped: dict[str, list[str]] = defaultdict(list)
    for gene in ordered_genes:
        grouped[union_find.find(gene)].append(gene)
    components = [
        (component_identifier(members), tuple(sorted(members)))
        for members in grouped.values()
    ]
    return sorted(components, key=lambda item: item[0])


def assign_components_to_folds(
    components: Sequence[tuple[str, Sequence[str]]],
    *,
    fold_count: int = FOLD_COUNT,
    seed: int = FOLD_SEED,
) -> dict[str, int]:
    if fold_count < 2:
        raise FamilyHoldoutError("fold_count must be at least two")
    fold_sizes = [0] * fold_count
    assignment: dict[str, int] = {}

    def tie_hash(component_id: str) -> str:
        return hashlib.sha256(f"{seed}\0{component_id}".encode("utf-8")).hexdigest()

    ordered = sorted(
        components,
        key=lambda item: (-len(item[1]), tie_hash(item[0]), item[0]),
    )
    for component_id, members in ordered:
        fold = min(range(fold_count), key=lambda index: (fold_sizes[index], index))
        fold_sizes[fold] += len(members)
        for gene in members:
            if gene in assignment:
                raise FamilyHoldoutError(f"gene assigned more than once: {gene}")
            assignment[gene] = fold
    return assignment


def validate_split(
    genes: Iterable[str],
    pairs: Sequence[PairEvidence],
    identity_percent: float,
    component_by_gene: Mapping[str, str],
    fold_by_gene: Mapping[str, int],
    duplicate_groups: Sequence[Sequence[str]],
) -> dict[str, int]:
    expected = set(genes)
    if set(component_by_gene) != expected or set(fold_by_gene) != expected:
        raise FamilyHoldoutError("component/fold mappings do not cover the gene universe")
    cross_component_edges = 0
    cross_fold_edges = 0
    qualifying_edges = 0
    for pair in pairs:
        if not pair.qualifies(identity_percent):
            continue
        qualifying_edges += 1
        if component_by_gene[pair.gene_a] != component_by_gene[pair.gene_b]:
            cross_component_edges += 1
        if fold_by_gene[pair.gene_a] != fold_by_gene[pair.gene_b]:
            cross_fold_edges += 1
    duplicate_fold_violations = 0
    for group in duplicate_groups:
        if len({fold_by_gene[gene] for gene in group}) != 1:
            duplicate_fold_violations += 1
    if cross_component_edges or cross_fold_edges or duplicate_fold_violations:
        raise FamilyHoldoutError(
            "split validation failed: "
            f"cross_component_edges={cross_component_edges}, "
            f"cross_fold_edges={cross_fold_edges}, "
            f"duplicate_fold_violations={duplicate_fold_violations}"
        )
    return {
        "qualifying_edges": qualifying_edges,
        "cross_component_qualifying_edges": cross_component_edges,
        "cross_fold_qualifying_edges": cross_fold_edges,
        "exact_duplicate_fold_violations": duplicate_fold_violations,
    }


def _tsv_bytes(header: Sequence[str], rows: Iterable[Sequence[object]]) -> bytes:
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(value) for value in row))
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_canonical_pairs(path: Path, pairs: Sequence[PairEvidence]) -> None:
    if path.exists():
        raise FamilyHoldoutError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    header = (
        "gene_a\tgene_b\tpident_percent\tcoverage_a\tcoverage_b\t"
        "alignment_length\tevalue\tbits\n"
    )
    try:
        with temporary.open("xb") as raw_handle:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_handle, mtime=0
            ) as gzip_handle:
                gzip_handle.write(header.encode("ascii"))
                for pair in pairs:
                    line = (
                        f"{pair.gene_a}\t{pair.gene_b}\t"
                        f"{pair.pident_percent:.6f}\t{pair.coverage_a:.6f}\t"
                        f"{pair.coverage_b:.6f}\t{pair.alignment_length}\t"
                        f"{pair.evalue:.12g}\t{pair.bits:.6f}\n"
                    )
                    gzip_handle.write(line.encode("utf-8"))
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _wsl_path(path: Path) -> str:
    """Translate an absolute Windows path for a pinned Linux binary under WSL."""
    completed = subprocess.run(
        ["wsl.exe", "-e", "wslpath", "-a", str(path.resolve())],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise FamilyHoldoutError(
            f"could not translate Windows path for WSL: {path}: "
            f"{completed.stderr[-2000:]}"
        )
    return completed.stdout.strip()


def _mmseqs_execution_paths(
    mmseqs: Path,
    fasta: Path,
    raw_hits: Path,
    temporary_directory: Path,
) -> tuple[list[str], str, str, str]:
    """Return the executable prefix and data paths in the runtime namespace."""
    if os.name == "nt" and mmseqs.suffix.lower() != ".exe":
        return (
            ["wsl.exe", "-e", _wsl_path(mmseqs)],
            _wsl_path(fasta),
            _wsl_path(raw_hits),
            _wsl_path(temporary_directory),
        )
    return [str(mmseqs)], str(fasta), str(raw_hits), str(temporary_directory)


def _remove_tree(path: Path, allowed_parent: Path) -> None:
    """Remove an MMseqs scratch tree without following paths outside its parent."""
    resolved_path = path.resolve()
    resolved_parent = allowed_parent.resolve()
    if resolved_path.parent != resolved_parent or resolved_path == resolved_parent:
        raise FamilyHoldoutError(
            f"refusing to remove path outside the expected parent: {resolved_path}"
        )
    if not path.exists():
        return
    if os.name == "nt":
        completed = subprocess.run(
            ["wsl.exe", "-e", "rm", "-rf", "--", _wsl_path(path)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0 or path.exists():
            raise FamilyHoldoutError(
                f"could not remove MMseqs scratch directory {path}: "
                f"{completed.stderr[-2000:]}"
            )
        return
    shutil.rmtree(path)


def _run_mmseqs(
    mmseqs: Path,
    fasta: Path,
    raw_hits: Path,
    temporary_directory: Path,
    threads: int,
) -> tuple[list[str], str, str]:
    executable, runtime_fasta, runtime_hits, runtime_tmp = _mmseqs_execution_paths(
        mmseqs, fasta, raw_hits, temporary_directory
    )
    version_result = subprocess.run(
        [*executable, "version"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    version = version_result.stdout.strip()
    if version != MMSEQS_COMMIT:
        raise FamilyHoldoutError(
            f"MMseqs version {version!r} does not match pinned {MMSEQS_COMMIT}"
        )
    command = [
        *executable,
        "easy-search",
        runtime_fasta,
        runtime_fasta,
        runtime_hits,
        runtime_tmp,
        "--search-type",
        "1",
        "--min-seq-id",
        "0.20",
        "-c",
        "0.80",
        "--cov-mode",
        "0",
        "--alignment-mode",
        "3",
        "-e",
        "1e-5",
        "-s",
        "7.5",
        "--max-seqs",
        str(SEARCH_MAX_SEQS),
        "--threads",
        str(threads),
        "--format-output",
        MMSEQS_FIELDS,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise FamilyHoldoutError(
            "MMseqs search failed with exit code "
            f"{completed.returncode}: {completed.stderr[-4000:]}"
        )
    if not raw_hits.is_file() or raw_hits.stat().st_size == 0:
        raise FamilyHoldoutError("MMseqs search produced no non-empty output file")
    return command, version, completed.stderr[-4000:]


def _environment_manifest() -> dict[str, object]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "os_uname": list(os.uname()) if hasattr(os, "uname") else None,
    }


def _build_threshold_artifacts(
    root: Path,
    records: Sequence[SequenceRecord],
    pairs: Sequence[PairEvidence],
    identity_percent: float,
    fold_count: int,
    seed: int,
) -> dict[str, object]:
    genes = [record.gene for record in records]
    components = build_components(genes, pairs, identity_percent)
    fold_by_gene = assign_components_to_folds(
        components, fold_count=fold_count, seed=seed
    )
    component_by_gene = {
        gene: component_id
        for component_id, component_genes in components
        for gene in component_genes
    }
    duplicate_groups = exact_duplicate_groups(records)
    validation = validate_split(
        genes,
        pairs,
        identity_percent,
        component_by_gene,
        fold_by_gene,
        duplicate_groups,
    )
    record_by_gene = {record.gene: record for record in records}
    component_size = {
        component_id: len(component_genes)
        for component_id, component_genes in components
    }
    directory = root / f"identity_{int(identity_percent):02d}"
    directory.mkdir(parents=True, exist_ok=False)
    units_path = directory / "family_units.tsv"
    _atomic_write(
        units_path,
        _tsv_bytes(
            ("gene", "accession", "sequence_sha256", "component_id", "component_size"),
            (
                (
                    gene,
                    record_by_gene[gene].accession,
                    record_by_gene[gene].sequence_sha256,
                    component_by_gene[gene],
                    component_size[component_by_gene[gene]],
                )
                for gene in sorted(genes)
            ),
        ),
    )
    folds_path = directory / "fold_manifest.tsv"
    _atomic_write(
        folds_path,
        _tsv_bytes(
            ("gene", "component_id", "fold", "stratum"),
            (
                (
                    gene,
                    component_by_gene[gene],
                    fold_by_gene[gene],
                    "multi_gene_component"
                    if component_size[component_by_gene[gene]] > 1
                    else "singleton",
                )
                for gene in sorted(genes)
            ),
        ),
    )
    sizes = sorted((len(component_genes) for _, component_genes in components), reverse=True)
    fold_sizes = [sum(fold == index for fold in fold_by_gene.values()) for index in range(fold_count)]
    summary = {
        "schema": SCHEMA_VERSION,
        "identity_percent": identity_percent,
        "minimum_bidirectional_coverage": MIN_BIDIRECTIONAL_COVERAGE,
        "component_definition": "connected_components_of_qualifying_mmseqs_pairs",
        "assignment_algorithm": (
            "components_descending_size_then_sha256_seed_component_id; "
            "greedy_smallest_fold_then_lowest_fold_index"
        ),
        "fold_seed": seed,
        "fold_count": fold_count,
        "gene_count": len(genes),
        "component_count": len(components),
        "multi_gene_component_count": sum(size > 1 for size in sizes),
        "genes_in_multi_gene_components": sum(size for size in sizes if size > 1),
        "singleton_count": sum(size == 1 for size in sizes),
        "largest_component_size": sizes[0],
        "fold_gene_counts": fold_sizes,
        "exact_duplicate_groups": [list(group) for group in duplicate_groups],
        "validation": validation,
        "family_units": _file_record(units_path, root),
        "fold_manifest": _file_record(folds_path, root),
    }
    summary_path = directory / "split_summary.json"
    _atomic_write_json(summary_path, summary)
    return {
        "identity_percent": identity_percent,
        "summary": _file_record(summary_path, root),
        "component_count": len(components),
        "largest_component_size": sizes[0],
        "fold_gene_counts": fold_sizes,
        "qualifying_edges": validation["qualifying_edges"],
    }


def build_release(args: argparse.Namespace) -> Path:
    fasta = args.fasta.resolve()
    genes_path = args.genes.resolve()
    mmseqs = args.mmseqs.resolve()
    output = args.output.resolve()
    tool_archive = args.tool_archive.resolve() if args.tool_archive else None
    for required in (fasta, genes_path, mmseqs):
        if not required.is_file():
            raise FamilyHoldoutError(f"required file does not exist: {required}")
    if args.threads < 1:
        raise FamilyHoldoutError("threads must be positive")
    if output.exists():
        raise FamilyHoldoutError(f"refusing to overwrite output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        raise FamilyHoldoutError(
            f"staging directory already exists; inspect it before retrying: {staging}"
        )
    staging.mkdir()
    raw_hits = staging / "homology_hits.mmseqs.tsv"
    search_tmp = staging / "mmseqs_tmp"
    try:
        records = parse_fasta(fasta)
        expected_genes = read_expected_genes(genes_path)
        validate_gene_universe(records, expected_genes)
        command, version, stderr_tail = _run_mmseqs(
            mmseqs, fasta, raw_hits, search_tmp, args.threads
        )
        raw_record = {
            "bytes": raw_hits.stat().st_size,
            "sha256": sha256_file(raw_hits),
        }
        pairs, counts = parse_mmseqs_hits(raw_hits, expected_genes)
        validate_exact_duplicate_evidence(records, pairs)
        if any(
            pair.pident_percent + 1e-6 < min(REGISTERED_IDENTITIES)
            or pair.coverage_a + 1e-6 < MIN_BIDIRECTIONAL_COVERAGE
            or pair.coverage_b + 1e-6 < MIN_BIDIRECTIONAL_COVERAGE
            for pair in pairs
        ):
            raise FamilyHoldoutError(
                "canonical pair table violates the pinned search thresholds"
            )
        canonical_hits = staging / "homology_hits.tsv.gz"
        write_canonical_pairs(canonical_hits, pairs)
        _remove_tree(search_tmp, staging)
        raw_hits.unlink()

        threshold_records = [
            _build_threshold_artifacts(
                staging,
                records,
                pairs,
                identity,
                args.fold_count,
                args.seed,
            )
            for identity in REGISTERED_IDENTITIES
        ]
        source_manifest = {
            "schema": SCHEMA_VERSION,
            "analysis_code": {
                "path": Path(__file__).resolve().as_posix(),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "inputs": {
                "fasta": {
                    "path": fasta.as_posix(),
                    "bytes": fasta.stat().st_size,
                    "sha256": sha256_file(fasta),
                    "record_count": len(records),
                    "unique_sequence_count": len(
                        {record.sequence_sha256 for record in records}
                    ),
                },
                "genes": {
                    "path": genes_path.as_posix(),
                    "bytes": genes_path.stat().st_size,
                    "sha256": sha256_file(genes_path),
                    "gene_count": len(expected_genes),
                },
            },
            "tool": {
                "name": "MMseqs2",
                "release": MMSEQS_RELEASE,
                "reported_commit": version,
                "binary": {
                    "path": mmseqs.as_posix(),
                    "bytes": mmseqs.stat().st_size,
                    "sha256": sha256_file(mmseqs),
                },
                "release_archive": (
                    {
                        "path": tool_archive.as_posix(),
                        "bytes": tool_archive.stat().st_size,
                        "sha256": sha256_file(tool_archive),
                    }
                    if tool_archive and tool_archive.is_file()
                    else None
                ),
            },
            "search": {
                "command_argv": command,
                "registered_identity_percentages": list(REGISTERED_IDENTITIES),
                "main_identity_percentage": MAIN_IDENTITY,
                "minimum_bidirectional_coverage": MIN_BIDIRECTIONAL_COVERAGE,
                "coverage_mode": 0,
                "alignment_mode": 3,
                "evalue": SEARCH_EVALUE,
                "sensitivity": SEARCH_SENSITIVITY,
                "max_sequences": SEARCH_MAX_SEQS,
                "threads": args.threads,
                "format_output": MMSEQS_FIELDS,
                "raw_output": raw_record,
                "parse_counts": counts,
                "stderr_tail": stderr_tail,
            },
            "environment": _environment_manifest(),
            "canonical_hits": _file_record(canonical_hits, staging),
            "exact_duplicate_groups": [
                list(group) for group in exact_duplicate_groups(records)
            ],
        }
        _atomic_write_json(staging / "source_manifest.json", source_manifest)
        release_manifest = {
            "schema": SCHEMA_VERSION,
            "status": "complete",
            "scientific_use": "eligible_for_family_holdout_split_use",
            "main_identity_percentage": MAIN_IDENTITY,
            "thresholds": threshold_records,
            "files": [
                _file_record(path, staging)
                for path in sorted(staging.rglob("*"))
                if path.is_file() and path.name != "release_manifest.json"
            ],
        }
        _atomic_write_json(staging / "release_manifest.json", release_manifest)
        os.replace(staging, output)
        return output
    except Exception:
        # An incomplete staging directory is never a publishable result.
        if staging.exists():
            _remove_tree(staging, output.parent)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--tool-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--fold-count", type=int, default=FOLD_COUNT)
    parser.add_argument("--seed", type=int, default=FOLD_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        output = build_release(args)
    except (FamilyHoldoutError, OSError, subprocess.SubprocessError) as error:
        print(f"family holdout failed: {error}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
