"""candidate_annotation.py — Stage I addition: literature annotation of the 31
candidate targets via the OpenAlex API (real queries, real DOIs).

For each candidate: count of cancer-related publications (search "<GENE> cancer",
2015-2026) and the top-cited title; classify evidence status:
  well-characterized  (>=50 cancer papers)
  partially reported  (5-49)
  unreported          (<5)
Writes results/candidate_annotation.json + supplementary table S2 (markdown).
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def openalex_count(gene: str) -> tuple[int, str, str]:
    params = {
        "search": f'"{gene}" cancer',
        "filter": "from_publication_date:2015-01-01,type:article|review",
        "sort": "cited_by_count:desc",
        "per-page": "3",
        "mailto": "research@example.org",
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "DepPLM/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read().decode())
    except Exception as exc:
        return 0, "", f"API error: {exc}"
    count = payload.get("meta", {}).get("count", 0)
    results = payload.get("results", [])
    top_title = results[0].get("title", "") if results else ""
    top_doi = (results[0].get("doi") or "").replace("https://doi.org/", "") if results else ""
    return count, top_title, top_doi


def classify(n: int) -> str:
    if n >= 50:
        return "well-characterized"
    if n >= 5:
        return "partially reported"
    return "unreported"


def main() -> None:
    case = json.loads((RES / "cond_CASE.json").read_text(encoding="utf-8"))
    genes = case.get("candidate_list", [])
    print(f"[ann] annotating {len(genes)} candidates", flush=True)
    out = []
    for g in genes:
        n, title, doi = openalex_count(g)
        status = classify(n)
        out.append({"gene": g, "cancer_papers_2015": n, "top_paper": title,
                    "top_doi": doi, "status": status})
        print(f"[ann] {g}: {n} papers ({status})", flush=True)
        time.sleep(0.4)
    (RES / "candidate_annotation.json").write_text(json.dumps(out, indent=1),
                                                   encoding="utf-8")
    # Supplementary Table S2 (markdown)
    lines = ["| Gene | Cancer papers 2015- (OpenAlex) | Status | Top-cited paper (DOI) |",
             "|---|---|---|---|"]
    for r in sorted(out, key=lambda x: -x["cancer_papers_2015"]):
        lines.append(f"| {r['gene']} | {r['cancer_papers_2015']} | {r['status']} | "
                     f"{r['top_paper'][:80]} ({r['top_doi']}) |")
    (RES / "Supplementary_Table_S2.md").write_text("\n".join(lines), encoding="utf-8")
    n_novel = sum(1 for r in out if r["status"] == "unreported")
    n_partial = sum(1 for r in out if r["status"] == "partially reported")
    print(f"[ann] done: {n_novel} unreported, {n_partial} partially reported", flush=True)


if __name__ == "__main__":
    main()
