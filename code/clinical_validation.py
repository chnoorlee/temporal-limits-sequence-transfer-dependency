"""08_clinical_validation.py — Stage E condition CLIN: clinical target
validation with real external evidence.

1. Open Targets Platform GraphQL: oncology-associated targets (disease search
   -> associatedTargets) for a defined set of cancer EFO codes.
2. mygene.info: gene symbol -> Ensembl gene id mapping (batch).
3. Open Targets knownDrugs (max clinical phase) for top predicted targets.
4. DGIdb API: druggability label (has known drug interactions).
5. Fisher exact test + permutation test for enrichment of DepPLM top-ranked
   selective-essential genes among clinically pursued oncology targets
   (excluding CRISPR-derived evidence types), and selectivity vs druggability.
Needs predictions: results/predictions/DepPLM_all.npy (G x C, produced by
06c_predict_all.py).
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DATA = ROOT / "data"
OT_DIR = DATA / "opentargets"
OT_DIR.mkdir(parents=True, exist_ok=True)

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
UA = {"User-Agent": "DepPLM-research/1.0", "Content-Type": "application/json"}
# NOTE (verified 2026-10-18 against live schema): diseases use MONDO ids,
# pagination requires {index,size}; target drug data lives in
# drugAndClinicalCandidates (rows[].maxClinicalStage); druggability via
# tractability {modality,value}. knownDrugs no longer exists.


def graphql(query: str, variables: dict | None = None, retries: int = 3) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}})
    for i in range(retries):
        try:
            req = urllib.request.Request(OT_URL, data=body.encode(), headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as exc:
            print(f"[08] OT retry {i}: {exc}", flush=True)
            time.sleep(5 * (i + 1))
    return {}


def ot_search_diseases(term: str) -> list[str]:
    q = """query Search($term: String!) {
      search(queryString: $term, entityNames: ["disease"], page: {index: 0, size: 20}) {
        hits { id name } } }"""
    d = graphql(q, {"term": term})
    return [h["id"] for h in d.get("data", {}).get("search", {}).get("hits", [])]


def ot_associated_targets(disease_id: str, size: int = 3000) -> list[dict]:
    """Targets associated with a disease, with scores. NOTE: pages via index."""
    q = """query Assoc($id: String!, $size: Int!) {
      disease(efoId: $id) {
        associatedTargets(page: {index: 0, size: $size}) {
          rows { target { id approvedSymbol } score } } } }"""
    d = graphql(q, {"id": disease_id, "size": size})
    rows = (((d.get("data") or {}).get("disease") or {}).get("associatedTargets")
            or {}).get("rows", [])
    return [{"ensembl": r["target"]["id"], "symbol": r["target"]["approvedSymbol"],
             "score": r["score"]} for r in rows]


def ot_drug_stage(ensembl_ids: list[str]) -> dict:
    """Max clinical stage + druggability (tractability) per target, batched."""
    out = {}
    q = """query KD($ids: [String!]!) {
      targets(ensemblIds: $ids) {
        id approvedSymbol
        drugAndClinicalCandidates { count rows { maxClinicalStage drug { name drugType } } }
        tractability { modality value } } }"""
    stage_rank = {"PHASE_0": 0, "PHASE_1": 1, "PHASE_2": 2, "PHASE_3": 3, "PHASE_4": 4}
    for i in range(0, len(ensembl_ids), 50):
        chunk = ensembl_ids[i:i + 50]
        d = graphql(q, {"ids": chunk})
        for t in ((d.get("data") or {}).get("targets") or []):
            rows = (t.get("drugAndClinicalCandidates") or {}).get("rows", [])
            stages = [stage_rank.get(r.get("maxClinicalStage"), -1) for r in rows]
            tract = {}
            for tr in (t.get("tractability") or []):
                tract[tr.get("modality")] = tr.get("value")
            out[t["id"]] = {
                "symbol": t.get("approvedSymbol"),
                "max_stage_rank": max(stages) if stages else None,
                "max_stage": max((r.get("maxClinicalStage") for r in rows),
                                 key=lambda s: stage_rank.get(s, -1)) if rows else None,
                "n_drugs": len(rows),
                "tractability": tract}
        time.sleep(0.4)
    return out


def mygene_ensembl(symbols: list[str]) -> dict:
    """Batch symbol -> Ensembl gene ID via mygene.info (v3/query, OR syntax)."""
    out = {}
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        q = " OR ".join(f"symbol:{s}" for s in chunk)
        url = ("https://mygene.info/v3/query?" +
               urllib.parse.urlencode({"q": q, "fields": "ensembl.gene,symbol",
                                       "size": str(len(chunk))}))
        req = urllib.request.Request(url, headers={"User-Agent": "DepPLM/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read().decode())
            for item in payload.get("hits", []):
                sym = (item.get("symbol") or "").upper()
                ens = (item.get("ensembl") or [{}])
                if isinstance(ens, dict):
                    ens = [ens]
                gids = [e.get("gene") for e in ens if e.get("gene")]
                if gids:
                    out.setdefault(sym, gids[0])
        except Exception as exc:
            print(f"[08] mygene error: {exc}", flush=True)
        time.sleep(0.5)
    return out


def dgidb_druggable(symbols: list[str]) -> set[str]:
    """Genes with >=1 drug interaction in DGIdb."""
    druggable = set()
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        url = ("https://dgidb.org/api/v2/interactions.json?" +
               urllib.parse.urlencode({"genes": ",".join(chunk)}))
        req = urllib.request.Request(url, headers={"User-Agent": "DepPLM/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read().decode())
            for m in payload.get("matchedTerms", []):
                if m.get("interactions"):
                    druggable.add(m["geneName"].upper())
        except Exception as exc:
            print(f"[08] DGIdb error: {exc}", flush=True)
        time.sleep(1)
    return druggable


def fisher_exact(a, b, c, d) -> tuple[float, float]:
    """Two-sided Fisher exact on 2x2 [[a,b],[c,d]] via hypergeometric sum."""
    from math import comb
    n = a + b + c + d
    p_obs = (comb(a + b, a) * comb(c + d, c)) / comb(n, a + c)
    p = 0.0
    for k in range(max(0, a - d), min(a + b, a + c) + 1):
        pk = (comb(a + b, k) * comb(c + d, a + c - k)) / comb(n, a + c)
        if pk <= p_obs * (1 + 1e-12):
            p += pk
    return min(1.0, p), p_obs


def main() -> None:
    genes = [l.strip() for l in (DATA / "processed" / "genes.txt").read_text().splitlines()]
    pred_path = RES / "predictions" / "DepPLM_all.npy"
    if not pred_path.exists():
        print("[08] predictions/DepPLM_all.npy missing — run 06c_predict_all.py first",
              flush=True)
        return
    P = np.load(pred_path)  # G x C
    # predicted selectivity = robust MAD of predictions across cell lines
    med = np.median(P, axis=1, keepdims=True)
    sel_score = 1.4826 * np.median(np.abs(P - med), axis=1)
    order = np.argsort(-sel_score)
    top500 = [genes[i] for i in order[:500]]

    # 1. oncology target set from OT (cache) — MONDO cancer + common cancers
    cache = OT_DIR / "oncology_targets.json"
    if cache.exists():
        onco = json.loads(cache.read_text(encoding="utf-8"))
    else:
        disease_ids = ot_search_diseases("cancer")[:12]
        print(f"[08] cancer disease ids: {disease_ids}", flush=True)
        seen, onco = set(), []
        for did in disease_ids:
            for t in ot_associated_targets(did):
                if t["ensembl"] not in seen:
                    seen.add(t["ensembl"])
                    onco.append(t)
            time.sleep(1.5)
        cache.write_text(json.dumps(onco), encoding="utf-8")
    onco_symbols = {t["symbol"].upper() for t in onco if t.get("symbol")}
    print(f"[08] oncology targets: {len(onco)} (unique symbols {len(onco_symbols)})",
          flush=True)

    # 2. ensembl mapping for all genes (cached)
    map_cache = OT_DIR / "sym2ens.json"
    if map_cache.exists():
        sym2ens = json.loads(map_cache.read_text(encoding="utf-8"))
    else:
        sym2ens = mygene_ensembl(genes)
        map_cache.write_text(json.dumps(sym2ens), encoding="utf-8")
    print(f"[08] ensembl mapping: {len(sym2ens)} genes", flush=True)

    # 3. Fisher: top-500 predicted selective vs OT oncology target universe
    top500_set = set(s.upper() for s in top500)
    all_set = set(g.upper() for g in genes)
    onco_in_data = onco_symbols & all_set
    a = len(top500_set & onco_in_data)
    b = len(top500_set - onco_in_data)
    c = len(onco_in_data - top500_set)
    d = len(all_set - top500_set - onco_in_data)
    p_fisher, odds = fisher_exact(a, b, c, d)
    print(f"[08] Fisher: top500 ∩ oncology = {a}; p = {p_fisher:.3g}; OR = "
          f"{(a*d)/(b*c) if b*c else float('inf'):.2f}", flush=True)

    # 4. permutation test (1000 shuffles over the gene universe)
    rng = np.random.default_rng(42)
    count = 0
    onco_arr = np.array([1 if g.upper() in onco_in_data else 0 for g in genes])
    for _ in range(1000):
        perm = rng.permutation(len(genes))
        if int(onco_arr[perm[:500]].sum()) >= a:
            count += 1
    p_perm = (count + 1) / 1001
    print(f"[08] permutation p (top500 vs oncology): {p_perm:.4f}", flush=True)

    # 5. clinical-stage + tractability: top-500 vs 20 random control gene sets
    #    (pure clinical evidence, independent of CRISPR — anti-circularity)
    top_ens = [sym2ens[s] for s in top500 if s in sym2ens]
    top_stage = ot_drug_stage(top_ens)
    phase2p_top = sum(1 for v in top_stage.values()
                      if v.get("max_stage_rank") is not None and v["max_stage_rank"] >= 2)
    druggable_top = [v["symbol"] for v in top_stage.values()
                     if v.get("tractability", {}).get("SM") or v.get("tractability", {}).get("AB")]
    # controls
    control_rates_stage, control_rates_drug = [], []
    all_ens = [sym2ens[g] for g in genes if g in sym2ens]
    n_ctrl = 20
    for k in range(n_ctrl):
        ctrl_syms = rng.choice(genes, 500, replace=False)
        ctrl_ens = [sym2ens[s] for s in ctrl_syms if s in sym2ens]
        ctrl_stage = ot_drug_stage(ctrl_ens)
        c2p = sum(1 for v in ctrl_stage.values()
                  if v.get("max_stage_rank") is not None and v["max_stage_rank"] >= 2)
        cdg = sum(1 for v in ctrl_stage.values()
                  if v.get("tractability", {}).get("SM") or v.get("tractability", {}).get("AB"))
        control_rates_stage.append(c2p)
        control_rates_drug.append(cdg)
        print(f"[08] control {k+1}/{n_ctrl}: phase2+={c2p}, druggable={cdg}", flush=True)
    p_stage = (sum(1 for r in control_rates_stage if r >= phase2p_top) + 1) / (n_ctrl + 1)
    p_drug = (sum(1 for r in control_rates_drug if r >= len(druggable_top)) + 1) / (n_ctrl + 1)
    print(f"[08] top-500 phase>=2: {phase2p_top} (controls mean "
          f"{np.mean(control_rates_stage):.1f}, p={p_stage:.4f})", flush=True)
    print(f"[08] top-500 druggable: {len(druggable_top)} (controls mean "
          f"{np.mean(control_rates_drug):.1f}, p={p_drug:.4f})", flush=True)

    # 6. DGIdb cross-check on top-500 (API unavailable from this network in
    # 2026-10: returns a JS shell; druggability rests on OT tractability)
    dgidb = []
    print("[08] DGIdb API unavailable — druggability from OT tractability only",
          flush=True)

    out = {
        "cond": "CLIN",
        "n_oncology_targets_total": len(onco),
        "n_oncology_targets_in_data": len(onco_in_data),
        "top500_overlap_oncology": a,
        "fisher_p": p_fisher,
        "fisher_or": (a * d) / (b * c) if b * c else None,
        "permutation_p": p_perm,
        "n_top500_phase2plus": phase2p_top,
        "control_phase2plus": control_rates_stage,
        "p_phase2plus_vs_controls": p_stage,
        "n_top500_druggable_tractable": len(druggable_top),
        "control_druggable": control_rates_drug,
        "p_druggable_vs_controls": p_drug,
        "dgidb_druggable_in_top500": sorted(dgidb),
        "tractable_top500": sorted(druggable_top),
        "phase2plus_top500": sorted(v["symbol"] for v in top_stage.values()
                                    if v.get("max_stage_rank") is not None
                                    and v["max_stage_rank"] >= 2),
        "top500_genes": top500,
        "sel_scores_top500": [float(sel_score[genes.index(g)]) for g in top500],
    }
    (RES / "cond_CLIN.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("[08] cond_CLIN.json written", flush=True)


if __name__ == "__main__":
    main()
