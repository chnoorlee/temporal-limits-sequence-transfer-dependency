"""Download and inventory the public inputs used by this study.

The DepMap inputs are pinned to the DepMap 24Q4 Public Figshare release. Every
listed local file is size-checked when an authoritative byte count is known and
hashed even when it was already present before this script was run. The tracked
UniProt FASTA was acquired from release 2026_03; the provider URL is mutable, so
its exact bytes remain checksum-gated and require a public study archive before
submission.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEPMAP_DIR = DATA / "depmap"
UNIPROT_DIR = DATA / "uniprot"
STRING_DIR = DATA / "string"
for directory in (DEPMAP_DIR, UNIPROT_DIR, STRING_DIR):
    directory.mkdir(parents=True, exist_ok=True)

USER_AGENT = {"User-Agent": "DepPLM-research/1.0 (bioinformatics study)"}
DEPMAP_RELEASE_URL = (
    "https://plus.figshare.com/articles/dataset/DepMap_24Q4_Public/27993248"
)
DEPMAP_RELEASE_DOI = "10.25452/figshare.plus.27993248.v1"
DEPMAP_FILES = {
    "CRISPRGeneEffect.csv": {
        "url": "https://ndownloader.figshare.com/files/51064667",
        "size_bytes": 428_678_699,
        "sha256": "3d8f3ec6dbf2db7ff834b79b508622ec0b226f3518003fe96ecf5a4fcf167e3b",
    },
    "Model.csv": {
        "url": "https://ndownloader.figshare.com/files/51065297",
        "size_bytes": 645_696,
        "sha256": "b7a0c1385e6cef30132b56aff61f1261d11e3f490490b355c430d32ee0dbdcfa",
    },
    "OmicsSomaticMutations.csv": {
        "url": "https://ndownloader.figshare.com/files/51065732",
        "size_bytes": 338_945_382,
        "sha256": "92e86ab6a8457ef99908daf2a2c82d9529320e23cd6c6b7adc128adc1e0f41c8",
    },
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv": {
        "url": "https://ndownloader.figshare.com/files/51065489",
        "size_bytes": 506_628_654,
        "sha256": "2a71dc94110efcc0221eae821bb93a9f03b54bea16f005818911a09d33383d56",
    },
    "OmicsAbsoluteCNGene.csv": {
        "url": "https://ndownloader.figshare.com/files/51065303",
        "size_bytes": 238_808_692,
        "sha256": "5f530c797bf12bce68cccc3e763b29491c368bbbced8971733bed807a13b8c13",
    },
}
UNIPROT_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/reference_proteomes/Eukaryota/UP000005640/"
    "UP000005640_9606.fasta.gz"
)
UNIPROT_RELEASE = "2026_03"
UNIPROT_RELEASE_DATE = "2026-09-02"
UNIPROT_SIZE_BYTES = 7_728_297
STRING_LINKS_URL = (
    "https://stringdb-downloads.org/download/protein.links.v12.0/"
    "9606.protein.links.v12.0.txt.gz"
)
STRING_INFO_URL = (
    "https://stringdb-downloads.org/download/protein.info.v12.0/"
    "9606.protein.info.v12.0.txt.gz"
)
UNIPROT_SHA256 = "64299d01e8f41fb6ef03cdf7ec051d3d4e5f041003b6b0fa75b09d439af79dce"
STRING_INFO_SHA256 = (
    "144de4b0d98c6a7dfde6ddc2591cf88657f27b989eadff4f501450c3ed1f0f1c"
)
STRING_FILTERED_SHA256 = (
    "9218c2e97534d6249c6f76e60c08ca65b5b00b7725bcd3e0cb140b8076ea650c"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_file(
    path: Path,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
) -> tuple[int, str]:
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise RuntimeError(f"{path} has {actual_size} bytes; expected {expected_size}")
    actual_sha256 = sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{path} has SHA-256 {actual_sha256}; expected {expected_sha256}"
        )
    return actual_size, actual_sha256


def download(
    url: str,
    destination: Path,
    chunk_size: int = 1 << 20,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> None:
    """Download to a partial file and atomically install it.

    A partial transfer is resumed only when the server acknowledges the Range
    request. Otherwise the partial file is replaced, avoiding duplicated bytes.
    """

    partial = destination.with_suffix(destination.suffix + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    if have and expected_size is not None and have >= expected_size:
        if have == expected_size:
            try:
                _validate_file(
                    partial,
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                )
            except RuntimeError:
                pass
            else:
                os.replace(partial, destination)
                return
        partial.unlink()
        have = 0
    headers = dict(USER_AGENT)
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        resumed = have > 0 and getattr(response, "status", None) == 206
        if resumed:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
            if match is None or int(match.group(1)) != have:
                raise RuntimeError(
                    "server returned an invalid Content-Range for resume: "
                    f"{content_range!r}; expected start {have}"
                )
        mode = "ab" if resumed else "wb"
        total = have if resumed else 0
        with partial.open(mode) as handle:
            while block := response.read(chunk_size):
                handle.write(block)
                total += len(block)
                if total % (32 * chunk_size) < chunk_size:
                    print(f"  {destination.name}: {total / 1e6:.1f} MB", flush=True)
    _validate_file(
        partial,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    )
    os.replace(partial, destination)


def ensure_file(
    destination: Path,
    url: str,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, str | int]:
    if not destination.exists():
        print(f"  downloading {destination.name}", flush=True)
        download(
            url,
            destination,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
    actual_size, actual_sha256 = _validate_file(
        destination,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    )
    return {
        "file": destination.relative_to(DATA).as_posix(),
        "url": url,
        "size_bytes": actual_size,
        "sha256": actual_sha256,
    }


def filter_string_links(
    output_path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> int:
    """Stream, validate, then install STRING links with score >= 700."""

    temporary = output_path.with_suffix(output_path.suffix + ".part")
    edge_count = 0
    request = urllib.request.Request(STRING_LINKS_URL, headers=USER_AGENT)
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            gzip.GzipFile(fileobj=response) as compressed,
            temporary.open("w", encoding="utf-8", newline="") as output,
        ):
            compressed.readline()
            output.write("protein1\tprotein2\tcombined_score\n")
            for raw_line in compressed:
                fields = raw_line.decode("utf-8").strip().split()
                if len(fields) == 3 and int(fields[2]) >= 700:
                    output.write("\t".join(fields) + "\n")
                    edge_count += 1
            output.flush()
            os.fsync(output.fileno())
        _validate_file(
            temporary,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return edge_count


def write_sources(records: list[dict[str, str | int]]) -> None:
    path = DATA / "SOURCES.md"
    temporary = path.with_suffix(".md.tmp")
    legacy_artifacts = (
        ("processed/esm2_t6_8M.npy", DATA / "processed" / "esm2_t6_8M.npy"),
        (
            "processed/esm2_t6_8M.names.txt",
            DATA / "processed" / "esm2_t6_8M.names.txt",
        ),
        ("processed/gene_seqs.fa", DATA / "processed" / "gene_seqs.fa"),
        ("../code/03_embed_proteins.py", ROOT / "code" / "03_embed_proteins.py"),
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# Input data provenance\n\n")
        handle.write("Inventory audited on 2026-10-20.\n\n")
        handle.write(
            "Scope: this inventory covers the DepMap, UniProt, and STRING inputs "
            "used by the corrected-v2 predictive evaluation. It does **not** freeze "
            "the legacy Open Targets, mygene.info, or DGIdb API responses. Results "
            "derived from those mutable clinical queries are therefore exploratory "
            "and are excluded from corrected-v2 manuscript claims unless dated "
            "response snapshots and their query metadata are added here.\n\n"
        )
        handle.write(
            "DepMap inputs are pinned to [DepMap 24Q4 Public]("
            f"{DEPMAP_RELEASE_URL}) (DOI: `{DEPMAP_RELEASE_DOI}`).\n\n"
        )
        handle.write("| Local file | Source URL | Size (bytes) | SHA-256 |\n")
        handle.write("|---|---|---:|---|\n")
        for record in records:
            handle.write(
                f"| `{record['file']}` | {record['url']} | "
                f"{record['size_bytes']} | `{record['sha256']}` |\n"
            )
        handle.write("\n")
        handle.write(
            "`string/string_hs_700.tsv` is a deterministic derived file: the "
            "STRING v12.0 human protein-links stream was filtered to "
            "`combined_score >= 700`. The raw links archive is not retained "
            "locally.\n\n"
        )
        handle.write(
            "The UniProt bytes were acquired from release "
            f"`{UNIPROT_RELEASE}` ({UNIPROT_RELEASE_DATE}) and revalidated against "
            "that then-current provider file on 2026-10-20. The provider URL uses "
            "the mutable `current_release` path. The tracked file and SHA-256 above "
            "fix the exact local bytes, but they must also be deposited in the "
            "study's public versioned archive before submission; a future download "
            "may differ and must not silently replace this inventory.\n"
        )
        handle.write("\n## Legacy sequence-embedding artifact\n\n")
        handle.write(
            "The corrected runner is configured to reuse the processed artifacts "
            "listed below when they are present. Their current bytes can be fixed "
            "here, but their historical generation was not accompanied by an "
            "immutable run manifest, exact Hugging Face model/tokenizer revision, "
            "or environment record. The file name and current repository script "
            "indicate ESM-2 t6, but do not prove the saved array's execution origin. "
            "The runner therefore records "
            "`artifact_origin_status=unverified_legacy_origin`.\n\n"
        )
        handle.write(
            "| Local file | Status | Size (bytes) | SHA-256 |\n"
            "|---|---|---:|---|\n"
        )
        for label, artifact_path in legacy_artifacts:
            if artifact_path.is_file():
                handle.write(
                    f"| `{label}` | present | {artifact_path.stat().st_size} | "
                    f"`{sha256(artifact_path)}` |\n"
                )
            else:
                handle.write(f"| `{label}` | not present |  |  |\n")
    os.replace(temporary, path)


def main() -> None:
    records: list[dict[str, str | int]] = []

    print("[01] DepMap 24Q4 Public (pinned Figshare files)", flush=True)
    for name, metadata in DEPMAP_FILES.items():
        records.append(
            ensure_file(
                DEPMAP_DIR / name,
                str(metadata["url"]),
                expected_size=int(metadata["size_bytes"]),
                expected_sha256=str(metadata["sha256"]),
            )
        )

    print("[02] UniProt human reference proteome UP000005640", flush=True)
    records.append(
        ensure_file(
            UNIPROT_DIR / "UP000005640_9606.fasta.gz",
            UNIPROT_URL,
            expected_size=UNIPROT_SIZE_BYTES,
            expected_sha256=UNIPROT_SHA256,
        )
    )

    print("[03] STRING v12.0 human protein information", flush=True)
    records.append(
        ensure_file(
            STRING_DIR / "9606.protein.info.v12.0.txt.gz",
            STRING_INFO_URL,
            expected_size=1_970_090,
            expected_sha256=STRING_INFO_SHA256,
        )
    )

    print("[04] STRING v12.0 links filtered at combined_score >= 700", flush=True)
    filtered_path = STRING_DIR / "string_hs_700.tsv"
    if not filtered_path.exists():
        edge_count = filter_string_links(
            filtered_path,
            expected_size=22_271_454,
            expected_sha256=STRING_FILTERED_SHA256,
        )
        print(f"  retained {edge_count} edges", flush=True)
    records.append(
        ensure_file(
            filtered_path,
            STRING_LINKS_URL,
            expected_size=22_271_454,
            expected_sha256=STRING_FILTERED_SHA256,
        )
    )

    write_sources(records)
    print(
        "[done] data/SOURCES.md inventories DepMap, UniProt, and STRING inputs",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[error] {error}", file=sys.stderr, flush=True)
        raise
