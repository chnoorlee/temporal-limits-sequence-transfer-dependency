"""lit_search.py — Stage B: real literature retrieval via OpenAlex API.

Queries 2021-2026 SOTA literature for the DepPLM topic (CRISPR essentiality
deep learning, protein language models, PPI-GNN, drug target discovery).
Saves verified records (id, doi, title, year, journal, cited_by, abstract,
authors) as JSONL. No fabricated entries — every record comes from the API.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "literature"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "DepPLM-research/1.0 (mailto:research@example.org)"}

QUERIES = {
    "crispr_essentiality_dl": "CRISPR gene essentiality deep learning cancer dependency",
    "protein_lm_target": "protein language model cancer drug target discovery",
    "synth_lethality_gnn": "synthetic lethality prediction graph neural network",
    "chronos_depmap": "Chronos CRISPR screen gene dependency inference",
    "pancan_target_ml": "pan-cancer therapeutic target discovery machine learning",
    "gnn_ppi_essential": "graph neural network protein interaction essential genes",
    "crispr_benchmark_2024": "benchmark deep learning CRISPR dependency prediction",
    "llm_biomed_target": "large language model drug target identification oncology",
}


def fetch(query: str, per_page: int = 25) -> list[dict]:
    """Query OpenAlex with 2021+ filter, sort by citations. Returns works list."""
    params = {
        "search": query,
        "filter": "from_publication_date:2021-01-01,type:article|preprint|review",
        "sort": "cited_by_count:desc",
        "per-page": str(per_page),
        "mailto": "research@example.org",
    }
    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("results", [])


def abstract_from_inverted(inv: dict | None) -> str:
    """Reconstruct abstract text from OpenAlex inverted index."""
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def main() -> None:
    all_records = []
    for key, query in QUERIES.items():
        print(f"[lit_search] query '{key}': {query}", flush=True)
        try:
            works = fetch(query)
        except Exception as exc:  # network/API failure is reported, not papered over
            print(f"[lit_search] ERROR query {key}: {exc}", file=sys.stderr, flush=True)
            continue
        for w in works:
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            rec = {
                "query_key": key,
                "openalex_id": w.get("id"),
                "doi": doi,
                "title": (w.get("title") or "").strip(),
                "year": w.get("publication_year"),
                "journal": (w.get("primary_location") or {}).get("source") and (
                    w.get("primary_location") or {}).get("source", {}).get("display_name"),
                "type": w.get("type"),
                "cited_by_count": w.get("cited_by_count"),
                "authors": [a["author"]["display_name"] for a in w.get("authorships", [])[:5]],
                "abstract": abstract_from_inverted(w.get("abstract_inverted_index")),
                "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            all_records.append(rec)
        time.sleep(0.5)  # polite rate limiting

    out_path = OUT_DIR / "openalex_candidates.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[lit_search] saved {len(all_records)} records -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
