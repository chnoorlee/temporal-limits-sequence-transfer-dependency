"""04_build_graph.py — Stage E: STRING PPI graph with co-mutation edge weights.

- Downloads STRING 9606.protein.info.v12.0.txt.gz (protein -> preferred gene name)
- Loads filtered edges (string_hs_700.tsv, combined_score >= 700)
- Maps ENSP -> gene symbol via protein.info; keeps genes present in DepMap set
- Edge weight = clipped pointwise mutual information of co-mutation in DepMap
  cell lines (both endpoints in the 5000-gene hotspot set), else weight 1
- Saves edge list + weights (npz), plus node list
"""
import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "processed"
STRING_DIR = DATA / "string"

INFO_URL = ("https://stringdb-downloads.org/download/protein.info.v12.0/"
            "9606.protein.info.v12.0.txt.gz")


def get_protein_info() -> dict[str, str]:
    """ENSP -> preferred gene symbol."""
    path = STRING_DIR / "9606.protein.info.v12.0.txt.gz"
    if not path.exists():
        print("[04] downloading STRING protein.info ...", flush=True)
        req = urllib.request.Request(INFO_URL, headers={"User-Agent": "DepPLM/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
    mapping = {}
    with gzip.open(path, "rt") as fh:
        next(fh)  # header
        for line in fh:
            parts = line.rstrip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]  # string_id -> preferred_name
    return mapping


def load_edges(info: dict[str, str], genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    gene_set = set(genes)
    ensp_by_gene: dict[str, str] = {}
    for ensp, sym in info.items():
        if sym in gene_set:
            ensp_by_gene.setdefault(sym, ensp)
    sym_by_ensp = {v: k for k, v in ensp_by_gene.items()}
    edges_file = STRING_DIR / "string_hs_700.tsv"
    if not edges_file.exists():
        sys.exit("[04] string_hs_700.tsv missing — run 01_download_data.py first")
    src, dst = [], []
    with edges_file.open() as fh:
        next(fh)
        for line in fh:
            p1, p2, score = line.rstrip().split("\t")
            g1, g2 = sym_by_ensp.get(p1), sym_by_ensp.get(p2)
            if g1 and g2:
                src.append(g1)
                dst.append(g2)
    return np.array(src, dtype=object), np.array(dst, dtype=object)


def comutation_weights(edges_src, edges_dst, genes, cell_mut, top_mut_names) -> np.ndarray:
    """PMI-based co-mutation weight, clipped to [0, 3]; default 1.0 elsewhere."""
    gene_to_row = {g: i for i, g in enumerate(genes)}
    hotspot = {g: i for i, g in enumerate(top_mut_names) if g in gene_to_row}
    # binary vectors for hotspot genes
    idx = np.array([gene_to_row[g] for g in hotspot], dtype=np.int64)
    mut_cols = np.array([hotspot[g] for g in hotspot], dtype=np.int64)
    M = cell_mut[:, mut_cols]  # C x H binary
    p = M.mean(0)
    pab = (M.T @ M) / M.shape[0]  # H x H joint probs
    pa_pb = np.outer(p, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((pab + 1e-3) / (pa_pb + 1e-3))
    pmi = np.clip(pmi, 0, 3)
    # edge lookup
    w = np.ones(len(edges_src), dtype=np.float32)
    row_lookup = {g: i for i, g in enumerate(genes)}
    hot_idx = {g: k for k, g in enumerate(hotspot)}
    for i in range(len(edges_src)):
        a, b = hot_idx.get(edges_src[i]), hot_idx.get(edges_dst[i])
        if a is not None and b is not None:
            w[i] = pmi[a, b]
    return w


def main() -> None:
    genes = [l.strip() for l in (OUT / "genes.txt").read_text(encoding="utf-8").splitlines()]
    info = get_protein_info()
    src, dst = load_edges(info, genes)
    print(f"[04] edges (both mapped): {len(src)}", flush=True)

    cell_mut = np.load(OUT / "cell_mut.npy")
    # hotspot gene names: recompute from depmap mutation file order is expensive;
    # instead rebuild the same top-5000 list as in 02 (deterministic):
    import pandas as pd
    model_ids = [l.strip() for l in (OUT / "cell_lines.txt").read_text(encoding="utf-8").splitlines()]
    mut = pd.read_csv(DATA / "depmap" / "OmicsSomaticMutations.csv",
                      usecols=["HugoSymbol", "ModelID"], dtype=str)
    mut = mut[mut["ModelID"].isin(model_ids)]
    top5000 = mut["HugoSymbol"].value_counts().index[:5000]

    w = comutation_weights(src, dst, genes, cell_mut, top5000)
    print(f"[04] co-mutation weights: mean {w.mean():.3f}, "
          f"weighted edges {int((w > 1.01).sum())}", flush=True)

    node_list = genes
    node_idx = {g: i for i, g in enumerate(node_list)}
    ei = np.stack([np.array([node_idx[g] for g in src]),
                   np.array([node_idx[g] for g in dst])], axis=1).astype(np.int64)
    np.savez(OUT / "ppi_graph.npz", edge_index=ei, edge_weight=w, nodes=node_list)
    (OUT / "graph_nodes.txt").write_text("\n".join(node_list) + "\n", encoding="utf-8")
    print(f"[04] graph saved: {len(node_list)} nodes, {ei.shape[0]} edges", flush=True)


if __name__ == "__main__":
    main()
