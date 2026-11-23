"""gatdep_baseline.py — GATDep (2025) external SOTA baseline, equal budget.

Faithful port of GeneDependencyGAT (Ming et al., J Transl Med 2025,
DOI 10.1186/s12967-025-07501-3): input Linear -> 512, two GATConv layers
(heads 2 then 1, LayerNorm + ReLU), global-mean-pool broadcast, node-level
regression head. GATConv semantics follow PyG defaults (attention softmax
over neighbors incl. self-loops, no edge weights, negative slope 0.2).

Evaluation protocol is IDENTICAL to the t6/t12 scale comparison
(run_special.run_t12): same 2,000-gene subset, same 70/30 cell-line split
(seed 42), same shared GRID (hidden x lr, wd=1e-5), tune 20 ep/patience 5,
final 40 ep/patience 8, seed 42, bf16 autocast, masked MSE, mean per-cell
Pearson. hidden is tuned over {128, 256} (the shared grid) instead of the
authors' default 64 — documented equal-budget adaptation.

Writes results/cond_GATDEP.json (real numbers only).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_experiments import GRID
from run_special import load_all

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"
GD = DATA / "gatdep"


class GATLayerBatch(nn.Module):
    """GATConv on a shared graph with per-sample node features (B, G, F)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int):
        super().__init__()
        assert out_dim % heads == 0
        self.heads = heads
        self.head_dim = out_dim // heads
        self.lin = nn.Linear(in_dim, out_dim)
        self.att_l = nn.Linear(in_dim, heads)
        self.att_r = nn.Linear(in_dim, heads)

    def forward(self, x: torch.Tensor, src: torch.Tensor,
                dst: torch.Tensor, G: int) -> torch.Tensor:
        B = x.shape[0]
        h = self.lin(x).view(B, G, self.heads, self.head_dim)      # B,G,H,D
        a = self.att_l(x)[:, src, :] + self.att_r(x)[:, dst, :]    # B,E,H
        a = F.leaky_relu(a, 0.2)
        off = (torch.arange(B, device=x.device) * G)[:, None]
        src_b = (src[None, :].expand(B, -1) + off).reshape(-1)     # B*E
        dst_b = (dst[None, :].expand(B, -1) + off).reshape(-1)     # B*E
        N = B * G
        a_flat = a.reshape(B * src.shape[0], self.heads)           # B*E,H
        amax = torch.zeros(N, self.heads, device=x.device, dtype=a.dtype)
        amax.index_reduce_(0, src_b, a_flat, reduce="amax", include_self=False)
        e = torch.exp(a_flat - amax[src_b])                         # B*E,H
        denom = torch.zeros(N, self.heads, device=x.device, dtype=e.dtype)
        denom.index_add_(0, src_b, e)
        alpha = e / denom[src_b].clamp(min=1e-8)                    # B*E,H
        h_flat = h.reshape(N, self.heads, self.head_dim)
        msg = h_flat[dst_b] * alpha.unsqueeze(-1)                   # B*E,H,D
        out = torch.zeros(N, self.heads, self.head_dim,
                          device=x.device, dtype=msg.dtype)
        out.index_add_(0, src_b, msg)
        return out.view(B, G, -1)


class GeneDependencyGAT(nn.Module):
    """Port of GATDep/GeneDependencyGAT.py (no torch_geometric dependency)."""

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.layer = nn.Linear(in_channels, 512)
        self.ln1 = nn.LayerNorm(512)
        self.gat1 = GATLayerBatch(512, hidden_channels * 2, heads=2)
        self.norm1 = nn.LayerNorm(hidden_channels * 2)
        self.gat2 = GATLayerBatch(hidden_channels * 2, hidden_channels, heads=1)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.lin = nn.Linear(hidden_channels * 2, 1)

    def forward(self, x: torch.Tensor, src: torch.Tensor,
                dst: torch.Tensor, G: int) -> torch.Tensor:
        x = F.relu(self.ln1(self.layer(x)))
        x = F.relu(self.norm1(self.gat1(x, src, dst, G)))
        x = F.relu(self.norm2(self.gat2(x, src, dst, G)))
        xg = x.mean(dim=1, keepdim=True).expand(-1, G, -1)          # pool+broadcast
        return self.lin(torch.cat([x, xg], dim=-1)).squeeze(-1)     # B,G


def per_cell_pearson(pred: np.ndarray, truth: np.ndarray) -> float:
    r_list = []
    m = np.isfinite(truth)
    for i in range(pred.shape[0]):
        mi = m[i]
        r_list.append(np.corrcoef(pred[i][mi], truth[i][mi])[0, 1]
                      if mi.sum() >= 10 else np.nan)
    r = np.array(r_list)
    return float(np.nanmean(r[r > -2]))  # drop pathological nan pairs


def train_eval_gatdep(model, Y, edge_index, tr_idx, va_idx, device,
                      run_tag, max_epochs, patience, lr, wd):
    """Mirrors run_experiments.train_eval (same opt/sched/amp/masking)."""
    model = model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    best_val, best_state, bad = float("inf"), None, 0
    Yt = torch.from_numpy(Y)
    src = torch.from_numpy(edge_index[0]).long().to(device)
    dst = torch.from_numpy(edge_index[1]).long().to(device)
    G = Y.shape[0]
    mem_t = torch.from_numpy(MEM).to(device)
    for epoch in range(max_epochs):
        model.train()
        perm = np.random.permutation(tr_idx)
        tot, nbs = 0.0, 0
        for b in range(0, len(perm), 64):
            ci = perm[b:b + 64]
            # per-sample node features: membership (G,S) * gsva (B,S)
            xb = torch.einsum("gs,bs->bgs", mem_t,
                              torch.from_numpy(GSVA[ci]).to(device)).float()
            yb = Yt[:, ci].t().to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(xb, src, dst, G)
            m = torch.isfinite(yb)
            diff = torch.where(m, pred - yb, torch.zeros_like(pred))
            loss = (diff ** 2).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(ci)
            nbs += len(ci)
        sched.step()
        model.eval()
        vloss, vn, vr = 0.0, 0, []
        with torch.inference_mode():
            for b in range(0, len(va_idx), 64):
                ci = va_idx[b:b + 64]
                xb = torch.einsum("gs,bs->bgs", mem_t,
                                  torch.from_numpy(GSVA[ci]).to(device)).float()
                yb = Yt[:, ci].t().to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(xb, src, dst, G)
                P = pred.float().cpu().numpy()
                T = yb.float().cpu().numpy()
                mm = np.isfinite(T)
                vloss += float(np.sum((P - T)[mm] ** 2))
                vn += int(mm.sum())
                vr.append(per_cell_pearson(P, T))
        val_loss = vloss / max(1, vn)
        vp = float(np.nanmean([r for r in vr if r == r]))
        if epoch % 5 == 0 or epoch == max_epochs - 1:
            print(f"[{run_tag}] ep {epoch}: train {tot / max(nbs, 1):.4f} "
                  f"val {val_loss:.4f} pearson {vp:.4f}", flush=True)
        if val_loss < best_val - 1e-4:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_val


def evaluate_gatdep(model, X, Y, edge_index, idx, device):
    model.eval()
    Yt = torch.from_numpy(Y)
    src = torch.from_numpy(edge_index[0]).long().to(device)
    dst = torch.from_numpy(edge_index[1]).long().to(device)
    G = Y.shape[0]
    preds = []
    with torch.inference_mode():
        for b in range(0, len(idx), 64):
            ci = idx[b:b + 64]
            xb = torch.einsum("gs,bs->bgs", torch.from_numpy(MEM).to(device),
                              torch.from_numpy(GSVA[ci]).to(device)).float()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(xb, src, dst, G)
            preds.append(pred.float().cpu().numpy())
    P = np.concatenate(preds, axis=0)
    T = Yt[:, idx].t().numpy()
    m = np.isfinite(T)
    diff = P - T
    mae = float(np.sum(np.abs(diff[m])) / m.sum())
    pearson = per_cell_pearson(P, T)
    sp_list = []
    for i in range(P.shape[0]):
        mi = m[i]
        if mi.sum() < 10:
            continue
        sp_list.append(np.corrcoef(np.argsort(np.argsort(P[i][mi])),
                                   np.argsort(np.argsort(T[i][mi])))[0, 1])
    spearman = float(np.mean(sp_list)) if sp_list else float("nan")
    return {"pearson": pearson, "spearman": spearman, "mae": mae}


MEM = None
GSVA = None


def main() -> None:
    global MEM, GSVA
    Y, mut, expr, feat, lin, emb, ei, ew, adj = load_all()
    MEM = np.load(GD / "membership_sub2k.npy").astype(np.float32)
    GSVA = np.load(GD / "gsva_scores.npy").astype(np.float32)
    sub = np.load(GD / "sub2k_gene_idx.npy")
    e_sub = np.load(GD / "edge_index_sub2k.npy")
    genesets = json.loads((GD / "genesets.json").read_text(encoding="utf-8"))
    Ysub = Y[sub]
    G = Ysub.shape[0]
    print(f"[gatdep] genes={G} sets={MEM.shape[1]} "
          f"edges={e_sub.shape[1]}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(42)
    perm = rng.permutation(Y.shape[1])
    tr_idx = perm[: int(0.7 * Y.shape[1])]
    va_idx = perm[int(0.7 * Y.shape[1]):]
    print(f"[gatdep] split: {len(tr_idx)} train / {len(va_idx)} val cells "
          f"(seed 42, identical to run_t12)", flush=True)

    best = None
    for g in GRID:
        torch.manual_seed(42)
        model = GeneDependencyGAT(MEM.shape[1], g["hidden"])
        m, vloss = train_eval_gatdep(model, Ysub, e_sub, tr_idx, va_idx,
                                     device, f"GATDEP-tune-h{g['hidden']}-lr{g['lr']}",
                                     max_epochs=20, patience=5,
                                     lr=g["lr"], wd=g["wd"])
        met = evaluate_gatdep(m, GSVA, Ysub, e_sub, va_idx, device)
        print(f"[gatdep] tune h={g['hidden']} lr={g['lr']}: "
              f"val pearson {met['pearson']:.4f}", flush=True)
        if best is None or met["pearson"] > best[0]:
            best = (met["pearson"], g)

    torch.manual_seed(42)
    model = GeneDependencyGAT(MEM.shape[1], best[1]["hidden"])
    m, _ = train_eval_gatdep(model, Ysub, e_sub, tr_idx, va_idx, device,
                             "GATDEP-final", max_epochs=40, patience=8,
                             lr=best[1]["lr"], wd=best[1]["wd"])
    met = evaluate_gatdep(m, GSVA, Ysub, e_sub, va_idx, device)
    print(f"[gatdep] FINAL: pearson {met['pearson']:.4f} "
          f"spearman {met['spearman']:.4f} mae {met['mae']:.4f}", flush=True)

    ref = json.loads((RES / "cond_C-PLM-MLP-t12.json").read_text(encoding="utf-8"))
    out = {
        "cond": "GATDEP-2k",
        "model_ref": "Ming et al. 2025, DOI 10.1186/s12967-025-07501-3",
        "n_genes": G,
        "n_genesets": MEM.shape[1],
        "n_edges": int(e_sub.shape[1]),
        "split": "70/30 cell-line, seed 42 (identical to run_t12)",
        "grid": best[1],
        "gatdep": met,
        "reference_plmmlp": {
            "t6": ref.get("plmmlp_t6"),
            "t12": ref.get("plmmlp_t12"),
        },
        "gene_set_sources": sorted(set(k.split(" ")[0] for k in genesets)),
    }
    (RES / "cond_GATDEP.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print("[gatdep] cond_GATDEP.json written", flush=True)


if __name__ == "__main__":
    main()
