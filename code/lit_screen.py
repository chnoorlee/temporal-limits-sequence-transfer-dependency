"""lit_screen.py — Stage B gate: relevance + quality screening of OpenAlex candidates.

Dedupes by DOI, scores relevance from title+abstract keywords (core topic vs
tangential), keeps records above threshold, prints shortlist table and saves
JSONL. Quality signal = cited_by_count.
"""
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "literature"
IN_PATH = DATA / "openalex_candidates.jsonl"
OUT_PATH = DATA / "shortlist.jsonl"

CORE_KW = {
    "essentiality": ["essential gene", "gene essentiality", "essential genes", "fitness gene"],
    "crispr": ["crispr", "dependency screen", "depmap", "chronos", "gene effect"],
    "cancer_target": ["cancer", "tumor", "tumour", "oncology", "therapeutic target",
                       "drug target", "vulnerabilit"],
    "plm": ["protein language model", "esm-2", "esm2", "prottrans", "protein embeddings",
            "sequence embedding", "esm atlas"],
    "gnn": ["graph neural network", "graph attention", "ppi network", "protein-protein interaction",
            "message passing", "gcn", "gat"],
    "synthetic_lethality": ["synthetic lethal", "paralog"],
    "transfer": ["transfer learning", "generaliz", "zero-shot", "unseen", "held-out", "cross-cell line"],
    "druggab": ["druggab", "target identification", "target prioritiz", "open targets", "cansar"],
}
NEG_KW = ["plant", "arabidopsis", "yeast", "s. cerevisiae", "bacteria", "e. coli",
          "covid", "sars", "virus", "vaccine", "drosophila", "zebrafish", "mouse model of alzheimer"]


def score(rec: dict) -> tuple[float, list[str]]:
    text = ((rec.get("title") or "") + " " + (rec.get("abstract") or "")).lower()
    neg = any(k in text for k in NEG_KW)
    if neg:
        return 0.0, ["negative_keyword"]
    hits, missing = [], []
    for group, kws in CORE_KW.items():
        if any(k in text for k in kws):
            hits.append(group)
    core = len(hits)
    # domain relevance: must be cancer/target related AND (essentiality/crispr OR plm/gnn)
    if "cancer_target" not in hits:
        return core * 0.1, ["no_cancer"]
    if "essentiality" not in hits and "crispr" not in hits and "plm" not in hits:
        return core * 0.1, ["no_method"]
    # quality: citations as weak prior, recency bonus
    cites = rec.get("cited_by_count") or 0
    year = rec.get("year") or 2021
    q = min(cites / 200.0, 1.0)
    recency = min(max((year - 2020) / 6.0, 0.0), 1.0)
    return round(0.5 * core / len(CORE_KW) + 0.3 * q + 0.2 * recency, 4), hits


def main() -> None:
    seen, rows = set(), []
    with IN_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            key = rec.get("doi") or rec.get("openalex_id")
            if not key or key in seen:
                continue
            seen.add(key)
            s, hits = score(rec)
            if s >= 0.35:
                rec["relevance_score"] = s
                rec["hit_groups"] = hits
                rows.append(rec)
    rows.sort(key=lambda r: -r["relevance_score"])
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[lit_screen] shortlist: {len(rows)} records (from {len(seen)} unique)")
    print(f"{'score':>6} {'yr':>4} {'cites':>6}  title")
    for r in rows[:40]:
        print(f"{r['relevance_score']:>6} {r['year']:>4} {str(r['cited_by_count']):>6}  "
              f"{(r['title'] or '')[:95]}")


if __name__ == "__main__":
    main()
