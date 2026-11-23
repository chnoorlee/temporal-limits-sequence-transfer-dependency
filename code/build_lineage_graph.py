"""build_lineage_graph.py — optimization 4: lineage-specific co-expression
networks from DepMap expression (local file, chunked reads to bound RAM).

For each lineage with >= 25 cell lines:
  - standardize log-TPM per gene within the lineage's cell lines
  - keep genes with lineage-internal variance >= 0.05
  - chunked correlation (blocks of 512 genes) -> edges with |r| >= 0.5,
    capped at the top 100k per lineage
Union graph: edge weight = number of supporting lineages (>= 2 to be kept
in the union for the transfer experiment; per-lineage files also saved).
Output: data/processed/lineage_graph.npz {edge_index (E,2), edge_weight (E),
support (E)} + data/processed/lineage_edges/<lineage>.npz
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "processed"
LIN_EDGES = OUT / "lineage_edges"
LIN_EDGES.mkdir(parents=True, exist_ok=True)

MIN_CELLS = 25
VAR_THR = 0.05
CORR_THR = 0.5
TOP_K = 100_000
BLOCK = 512


def gene_columns() -> tuple[list[str], list[str]]:
    """Map expression columns 'SYMBOL (ID)' to our gene order."""
    genes = [l.strip() for l in (OUT / "genes.txt").read_text(encoding="utf-8").splitlines()]
    header = pd.read_csv(DATA / "depmap" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
                         nrows=1)
    sym_of_col = {}
    for c in header.columns[1:]:
        sym = c.split(" (")[0]
        sym_of_col.setdefault(sym, c)
    cols = []
    for g in genes:
        if g in sym_of_col:
            cols.append(sym_of_col[g])
        else:
            cols.append(None)
    return genes, cols


def main() -> None:
    genes, cols = gene_columns()
    valid_cols = [c for c in cols if c is not None]
    col_to_gene = np.array([i for i, c in enumerate(cols) if c is not None],
                           dtype=np.int64)
    n_missing = sum(1 for c in cols if c is None)
    print(f"[lng] {len(genes)} genes, {n_missing} without expression column",
          flush=True)

    lineages = [l.strip() for l in (OUT / "lineages.txt").read_text(encoding="utf-8").splitlines()]
    cells = [l.strip() for l in (OUT / "cell_lines.txt").read_text(encoding="utf-8").splitlines()]
    uni = sorted(set(lineages))
    support = np.zeros((len(genes), len(genes)), dtype=np.int8)  # 2.5GB... NO
    # instead: dict of (i,j)->support via per-lineage edge lists
    from collections import defaultdict
    sup = defaultdict(int)

    for li in uni:
        idx = [k for k, l in enumerate(lineages) if l == li]
        if len(idx) < MIN_CELLS:
            continue
        # read only this lineage's rows (chunked over the whole file, keep rows)
        X = None
        reader = pd.read_csv(DATA / "depmap" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
                             index_col=0, chunksize=200)
        blocks = []
        want = set(cells[k] for k in idx)  # ModelID strings, not positions
        for chunk in reader:
            take = [k for k, v in enumerate(chunk.index) if v in want]
            if take:
                sub = chunk.iloc[take][valid_cols].to_numpy(dtype=np.float32)
                full = np.zeros((sub.shape[0], len(genes)), dtype=np.float32)
                full[:, col_to_gene] = sub
                blocks.append(full)
        X = np.concatenate(blocks, 0)  # n_cells x G
        # lineage-internal variance filter
        gvar = np.var(X, axis=0)
        keep = np.where(gvar >= VAR_THR)[0]
        Z = X[:, keep]
        mu = Z.mean(0, keepdims=True)
        sd = Z.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        Z = (Z - mu) / sd
        n = Z.shape[0]
        edges = []
        Gk = Z.shape[1]
        for a in range(0, Gk, BLOCK):
            Za = Z[:, a:a + BLOCK]
            for b in range(a, Gk, BLOCK):
                Zb = Z[:, b:b + BLOCK]
                C = (Za.T @ Zb) / n
                ia, ib = np.nonzero(np.abs(C) >= CORR_THR)
                for u, v in zip(ia, ib):
                    gu, gv = keep[a + u], keep[b + v]
                    if gu == gv:
                        continue
                    edges.append((gu, gv, float(C[u, v])))
        edges.sort(key=lambda e: -abs(e[2]))
        edges = edges[:TOP_K]
        print(f"[lng] {li}: {len(idx)} cells, {Gk} variable genes, "
              f"{len(edges)} edges kept", flush=True)
        e = np.array([[x[0], x[1]] for x in edges], dtype=np.int64)
        w = np.array([x[2] for x in edges], dtype=np.float32)
        fname = li.replace("/", "_").replace("\\", "_").replace(" ", "_")
        np.savez(LIN_EDGES / f"{fname}.npz", edge_index=e, edge_weight=w,
                 n_cells=len(idx), n_var_genes=Gk)
        for (u, v), val in zip(e, w):
            key = (int(u), int(v)) if u < v else (int(v), int(u))
            sup[key] += 1

    # union with support >= 2
    keep = [(k[0], k[1], c) for k, c in sup.items() if c >= 2]
    print(f"[lng] union edges (support>=2): {len(keep)}", flush=True)
    e = np.array([[x[0], x[1]] for x in keep], dtype=np.int64)
    s = np.array([x[2] for x in keep], dtype=np.float32)
    np.savez(OUT / "lineage_graph.npz", edge_index=e, edge_weight=s, support=s)
    print("[lng] lineage_graph.npz written", flush=True)


if __name__ == "__main__":
    main()
