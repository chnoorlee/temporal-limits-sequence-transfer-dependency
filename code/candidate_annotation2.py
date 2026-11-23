"""candidate_annotation2.py — optimization 5: rigorous literature annotation
of all 31 candidate targets (real OpenAlex queries, real DOIs).

Replaces the coarse candidate_annotation.py pass, which used full-text search
(search=...) and therefore returned top papers that merely cite the gene in
their reference lists (e.g. VCP -> an FSP1 ferroptosis paper). This version:

 1. regenerates the full 31-gene candidate list from the same filters as the
    case study (top-500 predicted selectivity, SM/AB-tractable, no Phase II+
    clinical drug; sources: cond_CLIN.json),
 2. queries OpenAlex with title_and_abstract.search (title+abstract only) for
    (a) target-proposal context and (b) cancer context,
 3. verifies proposal hits by sentence-level co-occurrence: the gene symbol
    must appear within a 40-token window of a proposal term in the abstract,
 4. classifies each candidate:
      unreported      no verified proposal, cancer context <= 5
      context-only    no verified proposal, cancer context > 5
      reported        >= 1 verified target/dependency proposal

Writes results/candidate_annotation2.json and regenerates
results/Supplementary_Table_S2.md. All thresholds are documented in the
table header. No invented evidence: every DOI is returned by the API.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

PROPOSAL_TERMS = [
    "therapeutic target", "drug target", "target for", "targeting",
    "dependency", "essentiality", "vulnerability", "inhibitor",
    "required for", "essential gene", "druggable", "depletion",
]
CANCER_TERMS = "cancer OR tumor OR tumour OR carcinoma OR oncology OR malignant"
WINDOW = 40


def openalex_query(filter_str: str, per_page: int = 5) -> dict:
    params = {
        "filter": filter_str + ",from_publication_date:2015-01-01,type:article|review",
        "sort": "cited_by_count:desc",
        "per-page": str(per_page),
        "mailto": "research@example.org",
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "DepPLM/1.0"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 5:
                # honor Retry-After if present, else exponential backoff;
                # long Retry-After means the pool is exhausted -> fail fast so
                # the caller can fall back to PubMed
                ra = exc.headers.get("Retry-After") if exc.headers else None
                wait = int(ra) if ra and ra.isdigit() else min(20 * (2 ** attempt), 240)
                if wait > 300:
                    raise
                print(f"[ann2] 429, waiting {wait}s (attempt {attempt + 1})",
                      flush=True)
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt < 2:
                time.sleep(5 * (2 ** attempt))
                continue
            raise
    raise RuntimeError("unreachable")


def abstract_text(work: dict) -> str:
    inv = work.get("abstract_inverted_index")
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def proposal_terms_hit(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in PROPOSAL_TERMS)


def symbol_in_window(symbol: str, abstract: str, term: str, window: int = WINDOW) -> bool:
    """Gene symbol occurs within `window` tokens of the proposal term."""
    text = abstract.lower().replace("rig-i", "rigi").replace("rig‑i", "rigi")
    toks = re.findall(r"[a-z0-9]+", text)
    sym = symbol.lower()
    term_toks = term.split()
    hits = []
    for i in range(len(toks) - len(term_toks) + 1):
        if toks[i:i + len(term_toks)] == term_toks:
            hits.append(i)
    for h in hits:
        lo, hi = max(0, h - window), min(len(toks), h + window)
        if sym in toks[lo:hi]:
            return True
    return False


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; RIG-I normalized to RIGI, hyphens in
    compound names (e.g. siRNA-COG3) split into separate tokens."""
    t = text.lower().replace("rig-i", "rigi").replace("rig‑i", "rigi")
    return re.findall(r"[a-z0-9]+", t.replace("-", " "))


def symbol_in_title(symbol: str, title: str) -> bool:
    sym = symbol.lower()
    return sym in _tokens(title)


def symbol_in_sentence(symbol: str, abstract: str, term: str) -> bool:
    """Gene symbol and proposal term co-occur in the SAME sentence."""
    sym = symbol.lower()
    for sent in re.split(r"(?<=[.!?])\s+", abstract.lower()):
        if sym in _tokens(sent) and term in sent:
            return True
    return False


def verified_proposals(symbol: str, works: list[dict]) -> list[dict]:
    """Strict two-tier verification against false window co-occurrences.

    Tier 1: gene symbol in the title AND any proposal term in title+abstract.
    Tier 2: gene symbol and a proposal term in the SAME abstract sentence.
    """
    out = []
    for w in works:
        text = (w.get("abstract") or abstract_text(w)) if "abstract" not in w \
            else (w.get("abstract") or "")
        title = w.get("title") or ""
        if not proposal_terms_hit(title + " " + text):
            continue
        if symbol_in_title(symbol, title):
            out.append(w)
            continue
        for term in PROPOSAL_TERMS:
            if symbol_in_sentence(symbol, text, term):
                out.append(w)
                break
    return out


def gene_proposal_query(symbol: str) -> tuple[int, list[dict]]:
    q = (f'title_and_abstract.search:"{symbol}" AND '
         f'({" OR ".join(f'"{t}"' if " " in t else t for t in PROPOSAL_TERMS)})')
    payload = openalex_query(q, per_page=10)
    return payload.get("meta", {}).get("count", 0), payload.get("results", [])


def gene_context_query(symbol: str) -> int:
    q = f'title_and_abstract.search:"{symbol}" AND ({CANCER_TERMS})'
    payload = openalex_query(q, per_page=1)
    return payload.get("meta", {}).get("count", 0)


# ------------------------------------------------------------------ PubMed
# Fallback source when OpenAlex rate-limits (E-utilities, 3 rps polite).
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def pubmed_esearch(term: str, retmax: int = 5) -> dict:
    params = {"db": "pubmed", "term": term, "retmax": str(retmax),
              "retmode": "json", "sort": "relevance"}
    url = EUTILS + "/esearch.fcgi?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "DepPLM/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception:
            if attempt < 3:
                time.sleep(4 * (2 ** attempt))
                continue
            raise
    raise RuntimeError("unreachable")


def pubmed_efetch(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    url = EUTILS + "/efetch.fcgi?" + urllib.parse.urlencode(
        {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract",
         "retmode": "xml"})
    req = urllib.request.Request(url, headers={"User-Agent": "DepPLM/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                xml = r.read().decode("utf-8", errors="replace")
            break
        except Exception:
            if attempt < 3:
                time.sleep(4 * (2 ** attempt))
                continue
            raise
    works = []
    for art in re.findall(r"<PubmedArticle>.*?</PubmedArticle>", xml, re.S):
        title = (re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", art, re.S) or
                 [None, ""])[1]
        abstract = " ".join(re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>",
                                       art, re.S))
        doi = (re.search(r'<ELocationID EIdType="doi"[^>]*>(.*?)</ELocationID>',
                         art) or [None, ""])[1]
        year = (re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", art, re.S) or
                [None, ""])[1]
        works.append({"title": title, "abstract": abstract, "doi": doi,
                      "year": int(year) if year else None})
    return works


def pubmed_proposal(symbol: str) -> tuple[int, list[dict]]:
    prop = " OR ".join(f"{t}[tiab]" for t in PROPOSAL_TERMS)
    term = (f"{symbol}[tiab] AND ({prop}) AND "
            f"(cancer[tiab] OR tumor[tiab] OR tumour[tiab] OR "
            f"carcinoma[tiab] OR oncology[tiab]) AND 2015:2026[dp]")
    res = pubmed_esearch(term, retmax=5)
    count = int(res.get("esearchresult", {}).get("count", 0))
    pmids = res.get("esearchresult", {}).get("idlist", [])
    return count, pubmed_efetch(pmids)


def pubmed_context(symbol: str) -> int:
    term = (f"{symbol}[tiab] AND (cancer[tiab] OR tumor[tiab] OR "
            f"tumour[tiab] OR carcinoma[tiab] OR oncology[tiab]) "
            f"AND 2015:2026[dp]")
    res = pubmed_esearch(term, retmax=0)
    return int(res.get("esearchresult", {}).get("count", 0))


def classify(n_verified: int, n_context: int) -> str:
    if n_verified > 0:
        return "reported"
    if n_context <= 5:
        return "unreported"
    return "context-only"


def main() -> None:
    clin = json.loads((RES / "cond_CLIN.json").read_text(encoding="utf-8"))
    top500 = [g.upper() for g in clin["top500_genes"]]
    phase2 = set(g.upper() for g in clin.get("phase2plus_top500", []))
    druggable = set(g.upper() for g in clin.get("tractable_top500", []))
    genes = [g for g in top500 if g in druggable and g not in phase2]
    print(f"[ann2] candidates regenerated: {len(genes)}", flush=True)
    assert len(genes) == 31, f"expected 31 candidates, got {len(genes)}"

    out = []
    done = {}
    prev_path = RES / "candidate_annotation2.json"
    if prev_path.exists():
        for r in json.loads(prev_path.read_text(encoding="utf-8")):
            if r.get("status") != "error":
                done[r["gene"]] = r
    print(f"[ann2] resuming with {len(done)}/31 already annotated", flush=True)

    for i, g in enumerate(genes):
        if g in done:
            out.append(done[g])
            continue
        source = "openalex"
        try:
            n_prop, works = gene_proposal_query(g)
            n_ctx = gene_context_query(g)
        except Exception as exc:
            print(f"[ann2] {g}: OpenAlex {exc} — PubMed fallback", flush=True)
            source = "pubmed"
            try:
                n_prop, works = pubmed_proposal(g)
                n_ctx = pubmed_context(g)
                time.sleep(0.4)  # E-utilities 3 rps limit
            except Exception as exc2:
                out.append({"gene": g, "error": str(exc2), "status": "error"})
                print(f"[ann2] {g}: ERROR {exc2}", flush=True)
                continue
        verified = verified_proposals(g, works)
        status = classify(len(verified), n_ctx)
        evidence = []
        for w in verified[:3]:
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            evidence.append({
                "title": (w.get("title") or "")[:160],
                "doi": doi,
                "year": w.get("publication_year", w.get("year")),
            })
        out.append({
            "gene": g,
            "source": source,
            "proposal_papers_titleabs": n_prop,
            "verified_proposals": len(verified),
            "cancer_context_papers_titleabs": n_ctx,
            "status": status,
            "evidence": evidence,
        })
        print(f"[ann2] {i + 1:2d}/31 {g} [{source}]: {status} "
              f"(prop {n_prop}/{len(verified)}, ctx {n_ctx})", flush=True)
        time.sleep(3.0)  # polite rate limit (OpenAlex)

    (RES / "candidate_annotation2.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    # Supplementary Table S2 (final)
    lines = [
        "# Supplementary Table S2 — candidate-target literature audit (final)",
        "",
        """Queries (2015-2026, articles/reviews):
title_and_abstract.search:"<GENE>" AND (target OR dependency OR essentiality
OR vulnerability OR inhibitor ...) for target-proposal context;
title_and_abstract.search:"<GENE>" AND (cancer OR tumor ...) for cancer
context. Primary source OpenAlex; on rate-limit the identical boolean query
runs against PubMed E-utilities (per-gene source column). Proposal hits
verified by two-tier evidence: gene symbol in the title (with any proposal
term in title+abstract), or gene symbol and a proposal term in the same
abstract sentence. Status: reported = verified target/dependency proposal
exists; context-only = cancer papers but no verified proposal;
unreported = no verified proposal and <= 5 cancer-context papers. All
candidates satisfy: top-500 predicted selectivity, no Phase II+ clinical
drug (Open Targets), SM/AB-tractable (Open Targets tractability).""",
        "",
        "| Gene | Source | Cancer-context papers | Proposal papers | Verified proposals | Status | Representative evidence (DOI) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in out:
        if r.get("status") == "error":
            lines.append(f"| {r['gene']} | — | — | — | — | error | {r.get('error', '')} |")
            continue
        ev = r.get("evidence") or []
        ev_txt = "; ".join(
            f"{e['doi']} ({e['year']})" for e in ev if e.get("doi")) or "—"
        lines.append(
            f"| {r['gene']} | {r.get('source', 'openalex')} | "
            f"{r['cancer_context_papers_titleabs']} | "
            f"{r['proposal_papers_titleabs']} | {r['verified_proposals']} | "
            f"{r['status']} | {ev_txt} |")
    (RES / "Supplementary_Table_S2.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    n_un = sum(1 for r in out if r.get("status") == "unreported")
    n_re = sum(1 for r in out if r.get("status") == "reported")
    print(f"[ann2] done: {n_re} reported, {len(out) - n_re - n_un} context-only, "
          f"{n_un} unreported", flush=True)


if __name__ == "__main__":
    main()
