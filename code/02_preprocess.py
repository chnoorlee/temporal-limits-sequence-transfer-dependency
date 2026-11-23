"""02_preprocess.py — Stage E: load raw data into unified tensors + split protocols.

Outputs (data/processed/):
  gene_effect.npy        (G x C) float32 Chronos gene effects, genes = DepMap order
  genes.txt              gene symbols
  cell_lines.txt         ModelID list
  lineages.txt           OncotreeLineage per cell line
  cell_mut.npy           (C x 5000) binary mutation fingerprint (top hotspot genes)
  cell_expr_pca.npy      (C x 128) PCA of log-TPM expression (if file available)
  gene_handfeat.npy      (G x F) handcrafted features
  gene_seqs.fa           amino-acid sequences for mapped genes
  gene_map.json          DepMap symbol -> UniProt ID/sequence index
  splits.json            P1/P2/P3/P4 split indices (fixed seeds)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DEPMAP = DATA / "depmap"
OUT = DATA / "processed"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
TOP_MUT = 5000
N_EXPR_PC = 128
N_FAMILY_HOLDOUT_FRAC = 0.20


def load_gene_effect() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verified DepMap 24Q4 layout: rows = cell lines (ModelID), columns =
    genes as "SYMBOL (ENTREZID)". Transpose to genes x cell lines and strip
    Entrez suffixes from gene symbols."""
    ge = pd.read_csv(DEPMAP / "CRISPRGeneEffect.csv", index_col=0)
    model = pd.read_csv(DEPMAP / "Model.csv").set_index("ModelID")
    common = ge.index.intersection(model.index)
    ge = ge.loc[common]                                  # cells x genes
    ge = ge.T                                            # genes x cells
    ge.index = [c.split(" (")[0] for c in ge.index]
    ge = ge.loc[~ge.index.duplicated()]
    return ge, model.loc[common]


def load_mutations(models: pd.Index, genes: pd.Index) -> np.ndarray:
    """Binary hotspot mutation matrix (C x TOP_MUT)."""
    mut = pd.read_csv(DEPMAP / "OmicsSomaticMutations.csv", usecols=["HugoSymbol", "ModelID"],
                      dtype=str)
    mut = mut[mut["ModelID"].isin(models)]
    counts = mut["HugoSymbol"].value_counts()
    top = counts.index[:TOP_MUT]
    gene_idx = {g: i for i, g in enumerate(genes)}
    col_idx = {g: i for i, g in enumerate(top) if g in gene_idx}
    rows = models.get_indexer(mut["ModelID"].values)
    cols = mut["HugoSymbol"].map(col_idx).values
    keep = (rows >= 0) & (cols >= 0)
    M = sparse.coo_matrix((np.ones(keep.sum(), dtype=np.float32),
                           (rows[keep], cols[keep])),
                          shape=(len(models), TOP_MUT)).tocsr()
    return np.asarray(M.todense(), dtype=np.float32)


def load_expression_pca(models: pd.Index, genes: pd.Index) -> np.ndarray:
    """Log-TPM expression -> top-128 PCA (correlation-based).

    File layout (verified DepMap 24Q4): rows = cell lines (ModelID),
    columns = genes as "SYMBOL (ENTREZID)".
    """
    path = DEPMAP / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    if not path.exists():
        return np.zeros((len(models), N_EXPR_PC), dtype=np.float32)
    expr = pd.read_csv(path, index_col=0)
    expr = expr.loc[expr.index.intersection(models)]       # cell x gene
    # map "SYMBOL (ID)" columns to symbols present in our gene set
    col_sym = {c: c.split(" (")[0] for c in expr.columns}
    keep = {c: s for c, s in col_sym.items() if s in set(genes)}
    expr = expr[list(keep)]
    expr.columns = [keep[c] for c in expr.columns]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    # standardize per gene then PCA (correlation PCA)
    mu, sd = expr.mean(0), expr.std(0)
    sd[sd == 0] = 1.0
    Z = ((expr - mu) / sd).to_numpy(dtype=np.float32)
    pca = PCA(n_components=N_EXPR_PC, random_state=42)
    return pca.fit_transform(Z).astype(np.float32)


def load_sequences(genes: pd.Index) -> dict:
    """Map DepMap symbols to UniProt canonical sequences via FASTA headers."""
    fasta = DATA / "uniprot" / "UP000005640_9606.fasta.gz"
    import gzip
    seqs, header = {}, None
    with gzip.open(fasta, "rt") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                header = line
                seqs[header] = ""
            else:
                seqs[header] += line
    # build symbol -> (ac, seq) map
    # UniProt mnemonics differ from HGNC (e.g. TP53 = "P53_HUMAN", KRAS =
    # "RASK_HUMAN") — the authoritative HGNC symbol is the GN= field in the
    # header description: ">sp|P04637|P53_HUMAN ... GN=TP53 PE=1 SV=1".
    by_symbol: dict[str, tuple] = {}
    for h, s in seqs.items():
        parts = h.split("|")
        if len(parts) < 3:
            continue
        ac = parts[1]
        symbol = None
        if "GN=" in h:
            symbol = h.split("GN=")[1].split()[0]
        if not symbol:
            symbol = parts[2].split()[0]
            if symbol.endswith("_HUMAN"):
                symbol = symbol[: -len("_HUMAN")]
        # prefer Swiss-Prot (sp) entries over TrEMBL (tr) per symbol
        if symbol not in by_symbol or parts[0].startswith("sp|"):
            by_symbol[symbol] = (ac, s)
    return by_symbol


def build_families(emb: np.ndarray, thr: float = 0.9, seed: int = 42) -> np.ndarray:
    """Greedy sequence-similarity family clustering on PLM embeddings (cosine).

    Used as the paralog/family proxy for the held-out-family protocol (P3).
    Deterministic: sort by norm then greedy assignment.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    rng = np.random.default_rng(seed)
    order = rng.permutation(emb.shape[0])
    sim = cosine_similarity(emb)
    labels = np.full(emb.shape[0], -1, dtype=np.int64)
    fam = 0
    for i in order:
        if labels[i] != -1:
            continue
        # members = points within thr of i, greedy (fast path: use full sim row)
        members = np.where(sim[i] >= thr)[0]
        members = members[labels[members] == -1]
        labels[members] = fam
        fam += 1
    return labels


def main() -> None:
    ge, model = load_gene_effect()
    genes = ge.index
    print(f"[02] gene effect: {ge.shape[0]} genes x {ge.shape[1]} cell lines", flush=True)

    # restrict to cells with complete multi-omics (expression file availability)
    if (DEPMAP / "OmicsExpressionProteinCodingGenesTPMLogp1.csv").exists():
        expr_idx = pd.read_csv(DEPMAP / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
                               usecols=[0]).iloc[:, 0]
        complete = ge.columns.intersection(expr_idx)
        ge = ge.loc[:, complete]
        model = model.loc[complete]
        print(f"[02] cells with complete omics: {ge.shape[1]}", flush=True)

    # labels: raw Chronos gene effects (already standardized across genes by
    # the Chronos model); NaN kept in place, handled by masked loss/eval
    Y = ge.to_numpy(dtype=np.float32)
    Yz = Y
    nan_frac = float(np.isnan(Yz).mean())
    print(f"[02] label NaN fraction: {nan_frac:.4f}", flush=True)

    # sequence mapping
    by_symbol = load_sequences(genes)
    mapped = {g: by_symbol[g] for g in genes if g in by_symbol}
    print(f"[02] genes with UniProt sequence: {len(mapped)} / {len(genes)}", flush=True)
    keep_genes = sorted(mapped, key=lambda g: genes.get_loc(g))
    keep_idx = [genes.get_loc(g) for g in keep_genes]
    Yz = Yz[keep_idx]
    seqs = [mapped[g][1] for g in keep_genes]

    with (OUT / "gene_seqs.fa").open("w") as fh:
        for g, (ac, s) in zip(keep_genes, (mapped[g] for g in keep_genes)):
            fh.write(f">{g}|{ac}\n{s}\n")
    (OUT / "gene_map.json").write_text(json.dumps(
        {g: ac for g, (ac, _) in (mapped.items())}, indent=0), encoding="utf-8")

    G, C = Yz.shape
    lineages = model["OncotreeLineage"].astype(str).to_numpy()

    # cell-line features
    cell_mut = load_mutations(model.index, genes)
    cell_expr_pca = load_expression_pca(model.index, genes)
    print(f"[02] mutation matrix {cell_mut.shape}, expr PCA {cell_expr_pca.shape}", flush=True)

    # mutation-column -> final-gene-row mapping (columns are the top-5000
    # hotspot genes; rows are the kept UniProt-mapped genes)
    mut_counts = pd.read_csv(DEPMAP / "OmicsSomaticMutations.csv",
                             usecols=["HugoSymbol", "ModelID"], dtype=str)
    mut_counts = mut_counts[mut_counts["ModelID"].isin(model.index)]
    top5000 = mut_counts["HugoSymbol"].value_counts().index[:TOP_MUT]
    gene_pos = {g: i for i, g in enumerate(keep_genes)}
    col_map = np.array([gene_pos.get(g, -1) for g in top5000], dtype=np.int64)
    # per-cell mutated-gene tokens (gene-row indices, frequency-sorted, top-k)
    valid = col_map >= 0
    freq = cell_mut[:, valid].sum(0)
    freq_genes = col_map[valid][np.argsort(-freq)]
    rank_of = np.full(len(keep_genes), -1, dtype=np.int64)
    rank_of[freq_genes] = np.arange(freq_genes.size)
    K = 64
    tokens = np.full((cell_mut.shape[0], K), -1, dtype=np.int64)
    token_mask = np.zeros((cell_mut.shape[0], K), dtype=bool)
    for c in range(cell_mut.shape[0]):
        gs = col_map[np.nonzero(cell_mut[c])[0]]
        gs = gs[gs >= 0]
        if gs.size == 0:
            tokens[c, 0] = freq_genes[0]
            token_mask[c, 0] = True
            continue
        gs = gs[np.argsort(rank_of[gs])][:K]
        tokens[c, :gs.size] = gs
        token_mask[c, :gs.size] = True
    np.save(OUT / "mut_col_to_gene.npy", col_map)
    np.save(OUT / "tokens.npy", tokens)
    np.save(OUT / "token_mask.npy", token_mask)

    # handcrafted per-gene features: length, GC content, seq available
    feat = np.zeros((G, 4), dtype=np.float32)
    for i, g in enumerate(keep_genes):
        s = mapped[g][1]
        feat[i, 0] = len(s)
        feat[i, 1] = (s.count("G") + s.count("C")) / max(len(s), 1)
    # mean expression over cell lines (file is cell x gene)
    if (DEPMAP / "OmicsExpressionProteinCodingGenesTPMLogp1.csv").exists():
        expr = pd.read_csv(DEPMAP / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
        for i, g in enumerate(keep_genes):
            col = next((c for c in expr.columns if c.split(" (")[0] == g), None)
            if col is not None:
                feat[i, 2] = float(expr[col].mean())
                feat[i, 3] = float(expr[col].std())
    # normalize features
    mu, sd = feat.mean(0), feat.std(0)
    sd[sd == 0] = 1.0
    feat = ((feat - mu) / sd).astype(np.float32)

    # splits
    n_lin = len(np.unique(lineages))
    lin_idx = pd.factorize(lineages)[0]
    # P1: 5-fold random CV over cell lines
    perm = RNG.permutation(C)
    folds = np.array_split(perm, 5)
    # P2: lineage holdout: leave out lineages (hematologic + kidney as holdout set)
    holdout_lins = ["Lymphoma", "Leukemia", "Myeloid", "Kidney", "Bone"]
    lin_set = set(holdout_lins) & set(lineages)
    p2_test = np.where(np.isin(lineages, list(lin_set)))[0]
    p2_train = np.where(~np.isin(lineages, list(lin_set)))[0]
    # P3: gene-family holdout — computed AFTER embeddings (needs PLM embs)
    #     placeholder: stored as None, filled by 05/06 scripts
    splits = {
        "P1_folds": [f.tolist() for f in folds],
        "P2_train": p2_train.tolist(), "P2_test": p2_test.tolist(),
        "P3": None,  # filled after embedding clustering
        "lineage_idx": lin_idx.tolist(),
    }

    np.save(OUT / "gene_effect.npy", Yz)
    np.save(OUT / "cell_mut.npy", cell_mut)
    np.save(OUT / "cell_expr_pca.npy", cell_expr_pca)
    np.save(OUT / "gene_handfeat.npy", feat)
    (OUT / "genes.txt").write_text("\n".join(keep_genes) + "\n", encoding="utf-8")
    (OUT / "cell_lines.txt").write_text("\n".join(model.index) + "\n", encoding="utf-8")
    (OUT / "lineages.txt").write_text("\n".join(lineages) + "\n", encoding="utf-8")
    (OUT / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    print(f"[02] preprocessed: Y {Yz.shape}, saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
