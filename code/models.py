"""models.py — model zoo for DepPLM experiments (PyTorch, no GNN framework).

- GCNConv: message passing on the pre-normalized weighted adjacency
  (index_add_ scatter-sum; CPU/GPU compatible)
- GATConv: edge-weighted multi-head attention with scatter softmax
- GeneHead: rank-r bilinear decoder y[g,b] = <Wz z_g, Wc c_b>/sqrt(r) + bias_g
  (O(G*r) per batch — makes full-genome training tractable)
- Baselines: FeatureMLP, OneHotMLP, PLMMLP, GNNRegressor
- DepPLM: PLM -> co-mutation GNN -> bilinear head + cross-attention over the
  cell line's mutated-gene tokens as an additive energy term; attention weights
  are not treated as faithful explanations;
  multitask selectivity head. Ablation switches for each claimed component.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- sparse graph
def normalized_adjacency(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                         n: int, self_loop: float = 1.0) -> torch.Tensor:
    """Return symmetric ``D^-1/2 (A + self_loop*I) D^-1/2``.

    Input edges are treated as records for an undirected graph. Repeated records,
    including an already-present reverse direction, are merged by their mean weight
    before the graph is symmetrized. This prevents a bidirectional edge table from
    being counted twice relative to the added self-loop.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weight must contain one value per edge record")
    if n < 1:
        raise ValueError("n must be positive")
    row, col = edge_index[0].long(), edge_index[1].long()
    if row.numel() and (int(torch.min(torch.minimum(row, col))) < 0 or
                        int(torch.max(torch.maximum(row, col))) >= n):
        raise IndexError("edge_index contains a node outside 0..n-1")
    vals = edge_weight.float()
    if not bool(torch.isfinite(vals).all()):
        raise ValueError("edge_weight must be finite")
    if bool((vals < 0).any()):
        raise ValueError("edge_weight must be non-negative")

    low = torch.minimum(row, col)
    high = torch.maximum(row, col)
    upper_indices = torch.stack([low, high])
    upper_sum = torch.sparse_coo_tensor(
        upper_indices, vals, (n, n), device=vals.device
    ).coalesce()
    upper_count = torch.sparse_coo_tensor(
        upper_indices, torch.ones_like(vals), (n, n), device=vals.device
    ).coalesce()
    unique_indices = upper_sum.indices()
    unique_values = upper_sum.values() / upper_count.values()

    unique_row, unique_col = unique_indices[0], unique_indices[1]
    off_diagonal = unique_row != unique_col
    r = torch.cat([
        unique_row,
        unique_col[off_diagonal],
        torch.arange(n, device=vals.device),
    ])
    c = torch.cat([
        unique_col,
        unique_row[off_diagonal],
        torch.arange(n, device=vals.device),
    ])
    v = torch.cat([
        unique_values,
        unique_values[off_diagonal],
        torch.full((n,), self_loop, dtype=vals.dtype, device=vals.device),
    ])
    A = torch.sparse_coo_tensor(
        torch.stack([r, c]), v, (n, n), device=vals.device
    ).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-8)
    d_inv = 1.0 / deg.sqrt()
    idx = A.indices()
    norm_vals = A.values() * d_inv[idx[0]] * d_inv[idx[1]]
    return torch.sparse_coo_tensor(idx, norm_vals, (n, n)).coalesce()


class GCNConv(nn.Module):
    """Message-passing GCN on the pre-normalized weighted adjacency."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        idx, val = adj.indices(), adj.values()
        src, dst = idx[0], idx[1]
        msg = x[src] * val.to(x.dtype).unsqueeze(-1)     # E x H
        out = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)
        return F.relu(self.lin(out))


class GATConv(nn.Module):
    """Edge-weighted GAT: attention + edge weight in log space."""
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.lin = nn.Linear(in_dim, out_dim)
        self.att_l = nn.Linear(in_dim, n_heads)
        self.att_r = nn.Linear(in_dim, n_heads)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h = self.lin(x).view(x.shape[0], self.n_heads, self.head_dim)  # G,H,D
        a = self.att_l(x)[src] + self.att_r(x)[dst]                     # E,H
        a = F.leaky_relu(a, 0.2)
        if edge_weight is not None:
            a = a + torch.log(edge_weight.clamp(min=1e-6)).to(a.dtype).unsqueeze(1)
        a_max = torch.zeros(x.shape[0], self.n_heads, device=x.device, dtype=a.dtype)
        a_max.index_reduce_(0, src, a, reduce="amax", include_self=False)
        e = torch.exp(a - a_max[src])                                   # E,H
        denom = torch.zeros(x.shape[0], self.n_heads, device=x.device, dtype=e.dtype)
        denom.index_add_(0, src, e)
        alpha = e / denom[src].clamp(min=1e-8)                          # E,H
        msg = h[dst] * alpha.unsqueeze(-1)                              # E,H,D
        out = torch.zeros(x.shape[0], self.n_heads, self.head_dim, device=x.device,
                          dtype=msg.dtype)
        out.index_add_(0, src, msg)
        return F.elu(out.view(x.shape[0], -1))


# ------------------------------------------------------------------- encoders
class CellEncoder(nn.Module):
    """Cell-line branch: mutation fingerprint + expr PCA + lineage one-hot."""
    def __init__(self, n_mut: int, n_expr: int, n_lin: int, out_dim: int = 128):
        super().__init__()
        self.fc_mut = nn.Linear(n_mut, out_dim)
        self.fc_expr = nn.Linear(n_expr, out_dim) if n_expr > 0 else None
        self.fc_lin = nn.Linear(n_lin, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, mut: torch.Tensor, expr: torch.Tensor, lin: torch.Tensor):
        z = F.relu(self.fc_mut(mut))
        if self.fc_expr is not None:
            z = z + F.relu(self.fc_expr(expr))
        z = z + F.relu(self.fc_lin(lin))
        return self.norm(z)


class GeneHead(nn.Module):
    """Rank-r bilinear decoder with gene-specific bias (shared by all models)."""
    def __init__(self, gene_dim: int, cell_dim: int, n_genes: int, rank: int = 64):
        super().__init__()
        self.rank = rank
        self.Wz = nn.Linear(gene_dim, rank, bias=False)
        self.Wc = nn.Linear(cell_dim, rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_genes))

    def forward(self, z_gene: torch.Tensor, c_cell: torch.Tensor,
                gene_idx: torch.Tensor | None = None):
        uz = self.Wz(z_gene)                       # G,r
        uc = self.Wc(c_cell)                       # B,r
        y = (uz @ uc.t()) / math.sqrt(self.rank)   # G,B
        b = self.bias
        if gene_idx is not None:
            b = self.bias[gene_idx.to(self.bias.device)]
        return (y + b.unsqueeze(1)).t()            # B,G


# ------------------------------------------------------------------- baselines
class DeepDepMLP(nn.Module):
    """C-DEEPDEP: DeepDEP-archetype baseline (Chiu et al., Sci Adv 2021).

    DeepDEP maps a multi-omics fingerprint of the cell line to its dependency
    profile through a deep fully connected encoder with dropout. Here the
    fingerprint is the mutation/expression/lineage stack; gene embeddings are
    learned from scratch and decoded with the shared bilinear head, under the
    same tuning budget as all other models."""
    def __init__(self, n_genes: int, n_mut: int, n_expr: int, n_lin: int,
                 gene_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(n_genes, gene_dim)
        self.cell_deep = nn.Sequential(
            nn.Linear(n_mut + n_expr + n_lin, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, gene_dim), nn.LayerNorm(gene_dim))
        self.head = GeneHead(gene_dim, gene_dim, n_genes)
        self.register_buffer("arange", torch.arange(n_genes))

    def forward(self, mut, expr, lin, gene_subset=None):
        x = torch.cat([mut, expr, lin], dim=1)
        c = self.cell_deep(x)
        if gene_subset is not None:
            idx = gene_subset.to(self.emb.weight.device)
            return self.head(self.emb(idx), c, gene_subset)
        return self.head(self.emb(self.arange), c)


class FeatureMLP(nn.Module):
    """C-FEAT: handcrafted gene features + cell branch."""
    def __init__(self, n_feat: int, n_mut: int, n_expr: int, n_lin: int,
                 n_genes: int, gene_dim: int = 128):
        super().__init__()
        self.gene_enc = nn.Sequential(nn.Linear(n_feat, gene_dim), nn.ReLU(),
                                      nn.Linear(gene_dim, gene_dim))
        self.cell = CellEncoder(n_mut, n_expr, n_lin, gene_dim)
        self.head = GeneHead(gene_dim, gene_dim, n_genes)

    def forward(self, feat, mut, expr, lin, gene_subset=None):
        if gene_subset is not None:
            return self.head(self.gene_enc(feat[gene_subset]),
                             self.cell(mut, expr, lin), gene_subset)
        return self.head(self.gene_enc(feat), self.cell(mut, expr, lin))


class OneHotMLP(nn.Module):
    """C-ONEHOT: learned gene embedding from scratch (no sequence prior)."""
    def __init__(self, n_genes: int, n_mut: int, n_expr: int, n_lin: int,
                 gene_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(n_genes, gene_dim)
        self.cell = CellEncoder(n_mut, n_expr, n_lin, gene_dim)
        self.head = GeneHead(gene_dim, gene_dim, n_genes)
        self.register_buffer("arange", torch.arange(n_genes))

    def forward(self, mut, expr, lin, gene_subset=None):
        if gene_subset is not None:
            idx = gene_subset.to(self.emb.weight.device)
            return self.head(self.emb(idx), self.cell(mut, expr, lin), gene_subset)
        return self.head(self.emb(self.arange), self.cell(mut, expr, lin))


class PLMMLP(nn.Module):
    """C-PLM-MLP: frozen PLM embeddings -> bilinear head (ESM4SL-style)."""
    def __init__(self, plm: torch.Tensor, n_mut: int, n_expr: int, n_lin: int,
                 gene_dim: int = 128, freeze: bool = True):
        super().__init__()
        self.emb = nn.Parameter(plm.clone(), requires_grad=not freeze)
        self.proj = nn.Sequential(nn.Linear(plm.shape[1], gene_dim), nn.ReLU(),
                                  nn.Linear(gene_dim, gene_dim))
        self.cell = CellEncoder(n_mut, n_expr, n_lin, gene_dim)
        self.head = GeneHead(gene_dim, gene_dim, plm.shape[0])

    def forward(self, mut, expr, lin, gene_subset=None):
        if gene_subset is not None:
            return self.head(self.proj(self.emb[gene_subset]),
                             self.cell(mut, expr, lin), gene_subset)
        return self.head(self.proj(self.emb), self.cell(mut, expr, lin))


class GNNRegressor(nn.Module):
    """C-GCN / C-GAT / C-GCN-comut: PLM emb -> GNN -> bilinear head."""
    def __init__(self, plm: torch.Tensor, adj: torch.Tensor, edge_index: torch.Tensor,
                 edge_weight: torch.Tensor | None, n_mut: int, n_expr: int, n_lin: int,
                 conv: str = "gcn", hidden: int = 128, freeze: bool = True):
        super().__init__()
        self.conv_name = conv
        self.emb = nn.Parameter(plm.clone(), requires_grad=not freeze)
        self.proj = nn.Linear(plm.shape[1], hidden)
        if conv == "gcn":
            self.conv1 = GCNConv(hidden, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        else:
            self.conv1 = GATConv(hidden, hidden)
            self.conv2 = GATConv(hidden, hidden)
        if edge_weight is not None:
            self.register_buffer("adj", adj)
            self.register_buffer("ew", edge_weight)
        else:
            # static (unweighted) normalized adjacency, built properly
            self.register_buffer("adj", normalized_adjacency(
                edge_index, torch.ones(edge_index.shape[1]), plm.shape[0]))
            self.ew = None
        self.register_buffer("ei", edge_index)
        self.cell = CellEncoder(n_mut, n_expr, n_lin, hidden)
        self.head = GeneHead(hidden, hidden, plm.shape[0])

    def encode_genes(self):
        h = self.proj(self.emb)
        if self.conv_name == "gcn":
            h = self.conv1(h, self.adj)
            h = self.conv2(h, self.adj)
        else:
            h = self.conv1(h, self.ei, self.ew)
            h = self.conv2(h, self.ei, self.ew)
        return h

    def forward(self, mut, expr, lin, gene_subset=None):
        z = self.encode_genes()
        if gene_subset is not None:
            return self.head(z[gene_subset], self.cell(mut, expr, lin), gene_subset)
        return self.head(z, self.cell(mut, expr, lin))


class DepPLM(nn.Module):
    """Full model: PLM -> co-mutation GNN -> bilinear head + cross-attention
    over the cell line's mutated-gene tokens as an additive energy term;
    multitask selectivity head. Attention weights are not treated as faithful
    explanations.

    Ablation switches record whether the supplied embedding was pretrained and
    enable the GNN, co-mutation weights, multitask head, and cross-attention.
    Embedding initialization and trainability are independent: ``freeze``
    controls the latter for both pretrained and random-initialized embeddings.
    """
    def __init__(self, plm: torch.Tensor, adj: torch.Tensor, edge_index: torch.Tensor,
                 edge_weight: torch.Tensor, n_mut: int, n_expr: int, n_lin: int,
                 hidden: int = 128, k_tokens: int = 64,
                 use_pretrained: bool = True, use_gnn: bool = True,
                 use_comut: bool = True, use_multitask: bool = True,
                 use_crossattn: bool = True, freeze: bool = True):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_comut = use_comut
        self.use_multitask = use_multitask
        self.use_crossattn = use_crossattn
        self.use_pretrained = use_pretrained
        self.k_tokens = k_tokens
        self.emb = nn.Parameter(plm.clone(), requires_grad=not freeze)
        self.proj = nn.Linear(plm.shape[1], hidden)
        if use_gnn:
            self.conv1 = GCNConv(hidden, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        self.register_buffer("adj", adj)
        self.register_buffer("adj_static", normalized_adjacency(
            edge_index, torch.ones(edge_index.shape[1]), plm.shape[0]))
        self.register_buffer("ei", edge_index)
        self.register_buffer("ew", edge_weight)
        self.cell = CellEncoder(n_mut, n_expr, n_lin, hidden)
        self.head = GeneHead(hidden, hidden, plm.shape[0])
        self.attn_scale = nn.Parameter(torch.tensor(1.0))
        if use_crossattn:
            self.wq = nn.Linear(hidden, hidden)
            self.wk = nn.Linear(hidden, hidden)
            self.pad_token = nn.Parameter(torch.zeros(1, hidden))
        if use_multitask:
            self.sel_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                          nn.Linear(hidden, 1))

    def encode_genes(self) -> torch.Tensor:
        z = self.proj(self.emb)
        if self.use_gnn:
            a = self.adj if self.use_comut else self.adj_static
            z = self.conv1(z, a)
            z = self.conv2(z, a)
        return z  # (G,H)

    def _cross_attention(self, z: torch.Tensor, token_idx: torch.Tensor,
                         token_mask: torch.Tensor):
        """Query = gene reprs, keys/values = the cell's mutated-gene tokens.
        Returns (energy, attention): energy[b,g] = sum_k attn * score."""
        B, K = token_idx.shape
        q = self.wq(z)                                        # G,H
        tok = token_idx.clamp(min=0)                          # B,K
        t = self.proj(self.emb[tok.view(-1)]).view(B, K, -1)  # B,K,H
        t = torch.where(token_mask.unsqueeze(-1), t, self.pad_token)
        kk = self.wk(t)                                       # B,K,H
        H = q.shape[-1]
        G = q.shape[0]
        kk2 = kk.reshape(B * K, H)
        scores = (q @ kk2.t()).reshape(G, B, K) / math.sqrt(H)  # G,B,K
        scores = scores.permute(1, 0, 2)                      # B,G,K
        scores = scores.masked_fill(~token_mask.unsqueeze(1), float("-inf"))
        all_masked = (~token_mask.unsqueeze(1)).all(-1, keepdim=True)
        scores = torch.where(all_masked.expand_as(scores),
                             torch.zeros_like(scores), scores)
        attn = torch.softmax(scores, dim=-1)                  # B,G,K
        # energy with 0-filled scores (0 * 0 = 0; avoids -inf * 0 = NaN)
        scores0 = torch.where(scores == float("-inf"), torch.zeros_like(scores),
                              scores)
        energy = (attn * scores0).sum(-1)                     # B,G
        return energy, attn

    def forward(self, mut, expr, lin, token_idx, token_mask, gene_subset=None):
        z = self.encode_genes()
        zq = z if gene_subset is None else z[gene_subset]
        c = self.cell(mut, expr, lin)                         # B,H
        y = self.head(zq, c, gene_subset)                     # B,G
        attn = None
        if self.use_crossattn:
            energy, attn = self._cross_attention(zq, token_idx, token_mask)
            y = y + self.attn_scale * energy
        sel = self.sel_head(z).squeeze(-1) if self.use_multitask else None
        return y, sel, attn
