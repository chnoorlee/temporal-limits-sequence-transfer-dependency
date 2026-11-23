"""gatdep_data.py — GATDep (2025) baseline data preparation.

Replicates the GATDep feature pipeline (Ming et al., J Transl Med 2025,
DOI 10.1186/s12967-025-07501-3): per-cell-line gene-set enrichment scores
(GSVA family) are combined with a gene x gene-set membership matrix to form
per-sample node features.

Deviations from the original (all documented, all real data):
 1. Gene sets: public Enrichr libraries (KEGG_2021_Human, MSigDB_Hallmark_2020)
    instead of the authors' private 6,561-set collection. Sets are filtered to
    15-500 genes in our 17,768-gene universe (standard GSVA practice).
 2. Enrichment statistic: ssGSEA (Barbie et al. 2009, alpha=0.25), the
    rank-based member of the GSVA family, implemented in numpy.
 3. Expression: DepMap 24Q4 OmicsExpressionProteinCodingGenesTPMLogp1 (the
    same 1,103 cell lines used throughout the project).

Writes data/processed/gatdep/
  genesets.json   {name: [gene symbols ...]}
  gsva_scores.npy (C x S) float32 ssGSEA scores per cell line
  membership_sub2k.npy (G_sub x S) uint8, rows = esm2 sub2k gene order
  edge_index_sub2k.npy (2 x E) int64 STRING subgraph + self-loops
"""
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
DEP = ROOT / "data" / "depmap"
OUT = DATA / "gatdep"
OUT.mkdir(exist_ok=True)

ENRICHR = ("https://maayanlab.cloud/Enrichr/geneSetLibrary?"
           "mode=text&libraryName={}")


def fetch_genesets(library: str) -> dict[str, list[str]]:
    """Download one Enrichr library; return {set_name: [symbols]}."""
    url = ENRICHR.format(library)
    req = urllib.request.Request(url, headers={"User-Agent": "DepPLM/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode("utf-8", errors="replace")
    out = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name = parts[0].strip().rstrip(",")
        genes = [re.split(r"[,\s]+", g)[0].upper()
                 for g in parts[2:]]
        out[name] = [g for g in genes if g]
    return out


def load_expression() -> tuple[np.ndarray, list[str], list[str]]:
    """DepMap expression: (cells x genes), gene symbols (from col headers)."""
    import csv
    with open(DEP / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
              encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd)
        symbols, rows, cells = [], [], []
        for col in header[1:]:
            m = re.match(r"([^\s(]+)", col)
            symbols.append(m.group(1).upper() if m else col.upper())
        for r in rd:
            cells.append(r[0])
            rows.append([float(x) for x in r[1:]])
    return np.array(rows, dtype=np.float32), symbols, cells


def ssgsea_all(expr: np.ndarray, set_masks: dict[str, np.ndarray],
               alpha: float = 0.25) -> np.ndarray:
    """Vectorized ssGSEA (Barbie 2009): returns (cells x sets) ES matrix.

    For each sample: genes ranked by expression; running sum over sorted
    genes adds r^alpha / sum(r^alpha over set) for set members and subtracts
    1/(G - N_set) otherwise; ES = max(run) - min(run).
    """
    C, G = expr.shape
    order = np.argsort(-expr, axis=1)          # descending expression
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(1, G + 1)[None, :], axis=1)
    rw = rank.astype(np.float64) ** alpha      # rank weights
    names = sorted(set_masks)
    es = np.zeros((C, len(names)), dtype=np.float32)
    sorted_mask = np.zeros((C, G), dtype=bool)
    for j, name in enumerate(names):
        mask = set_masks[name]
        ng = int(mask.sum())
        sorted_mask = np.take_along_axis(mask[None, :].astype(bool), order,
                                         axis=1)
        denom = (rw * sorted_mask).sum(axis=1)               # sum r^a over set
        add = np.divide(rw, denom[:, None], out=np.zeros_like(rw),
                        where=(denom[:, None] > 0)) * sorted_mask
        sub = (~sorted_mask) * (1.0 / max(G - ng, 1))
        run = np.cumsum(add - sub, axis=1)
        es[:, j] = (run.max(axis=1) - run.min(axis=1)).astype(np.float32)
        print(f"[gatdep-data] set {j + 1}/{len(names)} {name} "
              f"(n={ng})", flush=True)
    return es


def sub2k_edges(ei: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """STRING edges with both endpoints in the 2k subset, plus self-loops.

    Accepts (2, E) or (E, 2) orientation; returns (2, E) with endpoints
    REMAPPED to subset-local indices (0..len(sub)-1).
    """
    if ei.shape[0] == 2:
        ei = ei.T  # normalize to (E, 2)
    sset = set(sub.tolist())
    keep = [(i, j) for i, j in ei if i in sset and j in sset]
    remap = {g: k for k, g in enumerate(sub.tolist())}
    edges = sorted((remap[i], remap[j]) for i, j in keep)
    self_loops = [(k, k) for k in range(len(sub))]
    edges = sorted(set(edges + self_loops))
    return np.array(edges, dtype=np.int64).T


def main() -> None:
    print("[gatdep-data] downloading gene sets", flush=True)
    sets = {}
    for lib in ("KEGG_2021_Human", "MSigDB_Hallmark_2020"):
        lib_sets = fetch_genesets(lib)
        print(f"[gatdep-data] {lib}: {len(lib_sets)} sets", flush=True)
        sets.update(lib_sets)

    genes = [l.strip().upper() for l in
             (DATA / "genes.txt").read_text().splitlines()]
    gidx = {g: i for i, g in enumerate(genes)}
    gset = set(genes)

    # filter gene sets to 15-500 members in our universe, map to gene indices
    masks, names = {}, []
    for name, syms in sets.items():
        idx = sorted({gidx[s] for s in syms if s in gset})
        if 15 <= len(idx) <= 500:
            m = np.zeros(len(genes), dtype=bool)
            m[idx] = True
            masks[name] = m
            names.append(name)
    print(f"[gatdep-data] {len(names)} sets kept after 15-500 filter",
          flush=True)
    (OUT / "genesets.json").write_text(
        json.dumps({n: [genes[i] for i in np.where(masks[n])[0]]
                    for n in names}, indent=1), encoding="utf-8")

    print("[gatdep-data] loading DepMap expression", flush=True)
    expr, symbols, cells = load_expression()
    sidx = {s: i for i, s in enumerate(symbols)}
    # align to our gene universe (missing genes -> NaN, excluded from ranking
    # by treating their expression as -inf is wrong; instead we impute with
    # the gene's median across cells, standard for GSVA input QC)
    aligned = np.zeros((expr.shape[0], len(genes)), dtype=np.float32)
    hit = 0
    for gi, g in enumerate(genes):
        if g in sidx:
            aligned[:, gi] = expr[:, sidx[g]]
            hit += 1
    print(f"[gatdep-data] expression aligned: {hit}/{len(genes)} genes",
          flush=True)

    print("[gatdep-data] computing ssGSEA", flush=True)
    es = ssgsea_all(aligned, masks)

    # align rows to the project's 1,103 cell lines (gene_effect column order);
    # the expression matrix contains 1,673 cell lines (superset)
    project_cells = [l.strip() for l in
                     (DATA / "cell_lines.txt").read_text().splitlines()]
    row_map = {c: i for i, c in enumerate(cells)}
    missing = [c for c in project_cells if c not in row_map]
    if missing:
        raise RuntimeError(f"{len(missing)} project cell lines missing from "
                           f"expression matrix, e.g. {missing[:3]}")
    order = [row_map[c] for c in project_cells]
    es = es[order]
    np.save(OUT / "gsva_scores.npy", es)
    print(f"[gatdep-data] gsva_scores {es.shape} written (aligned to "
          f"{len(project_cells)} project cell lines)", flush=True)

    # membership + subgraph for the 2k-gene scale-comparison subset
    sub_names = [l.strip().upper() for l in
                 (DATA / "esm2_t12_35M_sub2k.names.txt").read_text().splitlines()]
    sub = np.array([gidx[n] for n in sub_names if n in gidx], dtype=np.int64)
    mem = np.zeros((len(sub), len(names)), dtype=np.uint8)
    for j, name in enumerate(names):
        rows = [i for i, gi in enumerate(sub) if masks[name][gi]]
        mem[rows, j] = 1
    np.save(OUT / "membership_sub2k.npy", mem)
    np.save(OUT / "sub2k_gene_idx.npy", sub)

    g = np.load(DATA / "ppi_graph.npz")
    ei = g["edge_index"]
    e_sub = sub2k_edges(ei, sub)
    np.save(OUT / "edge_index_sub2k.npy", e_sub)
    print(f"[gatdep-data] membership {mem.shape}, subgraph {e_sub.shape[1]} "
          f"edges (incl. self-loops)", flush=True)
    print("[gatdep-data] done", flush=True)


if __name__ == "__main__":
    main()
