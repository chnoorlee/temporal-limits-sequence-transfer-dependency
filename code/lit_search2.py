"""lit_search2.py — Stage B round 2: precision method-oriented retrieval.

Round 1 surfaced mostly wet-lab biology (citation-sorted). Round 2 uses
relevance sorting + method-specific queries on OpenAlex AND Semantic Scholar,
targeting the 2021-2026 computational SOTA (Chronos/Chronos-L, DeepDEP, ADaM,
SCREENER, SL GNN methods, ESM/PLM applications, DTI deep learning).
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "literature"
DATA.mkdir(parents=True, exist_ok=True)

QUERIES = {
    "deepdep": "DeepDEP deep learning predict cancer dependencies",
    "essentiality_ml": "machine learning predict gene essentiality cancer cell lines",
    "chronos2": "Chronos-L improved gene effect estimation dependency",
    "esm2": "ESM-2 protein language model structure prediction embeddings",
    "plm_function": "protein language models predict protein function deep learning",
    "gnn_dti": "graph neural network drug-target interaction prediction deep learning",
    "sl_dl": "deep learning synthetic lethality prediction cancer paralog",
    "transfer_dep": "transfer learning cancer dependency prediction unseen cell lines",
    "pancan_drugresp": "deep learning pan-cancer drug response cell lines attention",
    "target_prioritize": "computational cancer target prioritization multi-omics network",
    "foundation_biomed": "foundation model single-cell transcriptomics scGPT cancer",
    "gnn_ppi": "graph neural networks protein-protein interaction networks representation",
}


def openalex(query: str, per_page: int = 20) -> list[dict]:
    params = {
        "search": query,
        "filter": "from_publication_date:2021-01-01,type:article|preprint",
        "per-page": str(per_page),
        "mailto": "research@example.org",
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "DepPLM/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode()).get("results", [])


def semantic_scholar(query: str, limit: int = 20) -> list[dict]:
    params = {
        "query": query,
        "limit": str(limit),
        "year": "2021-",
        "fields": "title,abstract,year,externalIds,citationCount,authors,venue,url",
    }
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?"
           + urllib.parse.urlencode(params))
    req = urllib.request.Request(url, headers={"User-Agent": "DepPLM/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    out = []
    for p in payload.get("data", []):
        out.append({
            "ss_id": p.get("paperId"),
            "doi": (p.get("externalIds") or {}).get("DOI"),
            "arxiv": (p.get("externalIds") or {}).get("ArXiv"),
            "title": p.get("title"),
            "year": p.get("year"),
            "journal": (p.get("venue") or ""),
            "cited_by_count": p.get("citationCount"),
            "authors": [a.get("name") for a in p.get("authors", [])[:5]],
            "abstract": p.get("abstract"),
            "url": p.get("url"),
        })
    return out


def main() -> None:
    records = []
    for key, q in QUERIES.items():
        print(f"[lit2] OA '{key}': {q}", flush=True)
        try:
            for w in openalex(q):
                doi = (w.get("doi") or "").replace("https://doi.org/", "")
                records.append({
                    "source": "openalex", "query_key": key,
                    "openalex_id": w.get("id"), "doi": doi,
                    "title": (w.get("title") or "").strip(),
                    "year": w.get("publication_year"),
                    "journal": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "type": w.get("type"), "cited_by_count": w.get("cited_by_count"),
                    "authors": [a["author"]["display_name"] for a in w.get("authorships", [])[:5]],
                    "abstract": abstract_from_inverted(w.get("abstract_inverted_index")),
                    "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
        except Exception as exc:
            print(f"[lit2] OA error {key}: {exc}", flush=True)
        time.sleep(0.4)
        print(f"[lit2] S2 '{key}': {q}", flush=True)
        try:
            for p in semantic_scholar(q):
                p.update({"source": "semanticscholar", "query_key": key,
                          "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                records.append(p)
        except Exception as exc:
            print(f"[lit2] S2 error {key}: {exc}", flush=True)
        time.sleep(1.2)  # S2 rate limit ~1 rps unauthenticated

    out = DATA / "openalex_s2_round2.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[lit2] saved {len(records)} records -> {out}", flush=True)


def abstract_from_inverted(inv):
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


if __name__ == "__main__":
    main()
