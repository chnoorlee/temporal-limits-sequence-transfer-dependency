"""06_run_experiments.py — Stage E: run the 16-condition experiment matrix.

Real training: AdamW, cosine schedule, early stopping (patience) + strict
convergence gate (5 consecutive epochs with d(val_loss) < 1e-6).
Time guard: TIME_BUDGET (from --time-budget, default 5h) -> graceful save at 80%.
Pilot mode (--pilot): 200 genes x 200 cells, 1 epoch, prints TIME_ESTIMATE.
Every run writes results/cond_<id>.json + appends results/tuning_log.jsonl.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from models import (DepPLM, DeepDepMLP, FeatureMLP, GNNRegressor, OneHotMLP,
                    PLMMLP, normalized_adjacency)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

T0 = time.time()
TIME_BUDGET = 5 * 3600
G_SUBSAMPLE = 6000  # stochastic per-batch gene subsample (0 = off)


def time_guard() -> bool:
    """Return True when >= 80% of budget consumed (graceful shutdown)."""
    return (time.time() - T0) > 0.8 * TIME_BUDGET


# ------------------------------------------------------------------- loading
def load_data(pilot: bool = False):
    Y = np.load(DATA / "gene_effect.npy").astype(np.float32)          # G,C
    mut = np.load(DATA / "cell_mut.npy").astype(np.float32)           # C,5000
    expr = np.load(DATA / "cell_expr_pca.npy").astype(np.float32)     # C,128
    feat = np.load(DATA / "gene_handfeat.npy").astype(np.float32)     # G,4
    lineages = [l.strip() for l in (DATA / "lineages.txt").read_text().splitlines()]
    lin_codes = pd_factorize(lineages)
    emb = np.load(DATA / "esm2_t6_8M.npy").astype(np.float32)         # G,320
    g = np.load(DATA / "ppi_graph.npz")
    edge_index = torch.from_numpy(g["edge_index"].astype(np.int64)).t().contiguous()
    edge_weight = torch.from_numpy(g["edge_weight"].astype(np.float32))
    adj = normalized_adjacency(edge_index, edge_weight, emb.shape[0])
    if pilot:
        rng = np.random.default_rng(42)
        gi = rng.choice(Y.shape[0], 200, replace=False)
        ci = rng.choice(Y.shape[1], 200, replace=False)
        # subset the graph to the pilot genes (consistent subgraph)
        gset = set(gi.tolist())
        e = edge_index.numpy()
        keep = np.isin(e[0], gi) & np.isin(e[1], gi)
        e2, w2 = e[:, keep], edge_weight.numpy()[keep]
        remap = {int(old): new for new, old in enumerate(gi)}
        e2 = np.array([[remap[int(a)], remap[int(b)]] for a, b in e2.T]).T
        ei2 = torch.from_numpy(e2.astype(np.int64)).contiguous()
        ew2 = torch.from_numpy(w2.astype(np.float32))
        adj2 = normalized_adjacency(ei2, ew2, len(gi))
        return (Y[gi][:, ci], mut[ci], expr[ci], feat[gi], lin_codes[ci],
                emb[gi], ei2, ew2, adj2, ci, gi)
    return (Y, mut, expr, feat, lin_codes, emb, edge_index, edge_weight, adj,
            None, None)


def pd_factorize(vals):
    import pandas as pd
    return pd.factorize(np.asarray(vals))[0]


def build_tokens(mut: np.ndarray, k: int = 64, gi: np.ndarray | None = None):
    """Per-cell-line mutated-gene tokens — loaded from preprocessing
    (gene-row indices in FULL gene space, frequency-sorted, top-k).
    gi (optional): pilot/zero-shot gene subset -> remap token indices into
    the subset's row space; tokens for genes outside the subset are masked."""
    tokens = np.load(DATA / "tokens.npy").astype(np.int64)
    mask = np.load(DATA / "token_mask.npy").astype(bool)
    if gi is not None:
        pos = {int(g): i for i, g in enumerate(gi)}
        remapped = np.array([[pos.get(int(v), -1) for v in row] for row in tokens],
                            dtype=np.int64)
        bad = remapped < 0
        mask = mask & ~bad
        remapped[bad] = 0
        # rows that lost all tokens: attend to gene 0 of the subset
        empty = mask.sum(1) == 0
        remapped[empty, 0] = 0
        mask[empty, 0] = True
        tokens = remapped
    return tokens, mask


# ------------------------------------------------------------------- metrics
def pearson(x, y, axis=0):
    xm, ym = x - x.mean(axis, keepdims=True), y - y.mean(axis, keepdims=True)
    num = (xm * ym).sum(axis)
    den = np.sqrt((xm ** 2).sum(axis) * (ym ** 2).sum(axis)) + 1e-8
    return num / den


def spearman(x, y, axis=0):
    from scipy.stats import rankdata
    rx = np.apply_along_axis(rankdata, axis, x)
    ry = np.apply_along_axis(rankdata, axis, y)
    return pearson(rx, ry, axis)


def selectivity_labels(Y: np.ndarray, top_frac: float = 0.25) -> np.ndarray:
    """Per-gene selectivity: robust MAD across cell lines (NaN-aware);
    binary top-quartile."""
    med = np.nanmedian(Y, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(Y - med), axis=1)
    thr = np.nanquantile(mad, 1 - top_frac)
    lab = (mad >= thr).astype(np.float32)
    lab[np.isnan(mad)] = 0.0
    return lab, mad


# ------------------------------------------------------------------ training
def train_eval(model, cfg, Y, mut, expr, feat, lin_codes, emb,
               edge_index, edge_weight, adj, tokens, token_mask,
               tr_idx, va_idx, sel_y, device, run_tag, max_epochs=60,
               patience=10, pilot=False):
    """Returns best val metrics dict + per-cell-line predictions on va set."""
    model = model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    best_val, best_state, bad = float("inf"), None, 0
    n_lin = int(np.max(lin_codes)) + 1

    Yt = torch.from_numpy(Y)
    for epoch in range(max_epochs):
        if pilot and epoch >= 1:
            break
        if time_guard():
            print(f"[{run_tag}] TIME GUARD: stop at epoch {epoch}", flush=True)
            break
        model.train()
        perm = np.random.permutation(tr_idx)
        bs = 64
        tot = 0.0
        use_amp = device == "cuda"
        for b in range(0, len(perm), bs):
            ci = perm[b:b + bs]
            mb = torch.from_numpy(mut[ci]).to(device)
            eb = torch.from_numpy(expr[ci]).to(device)
            lb = torch.from_numpy(lin_codes[ci])
            lb = torch.zeros((lb.shape[0], n_lin), device=device).scatter_(1, lb.long().unsqueeze(1).to(device), 1.0)
            yb = Yt[:, ci].t().to(device)                       # B,G
            # stochastic gene subsampling per batch (6k genes) for tractability;
            # validation always uses the full gene set
            gsub = None
            if G_SUBSAMPLE and yb.shape[1] > G_SUBSAMPLE:
                gsub = torch.randperm(yb.shape[1], device="cpu")[:G_SUBSAMPLE]
                yb = yb[:, gsub]
            def _fw():
                if cfg["model"] == "deepdep":
                    return model(mb, eb, lb, gene_subset=gsub), None
                if cfg["model"] in ("featmlp",):
                    return model(torch.from_numpy(feat).to(device), mb, eb, lb,
                                 gene_subset=gsub), None
                elif cfg["model"] == "onehot":
                    return model(mb, eb, lb, gene_subset=gsub), None
                elif cfg["model"] == "plmmlp":
                    return model(mb, eb, lb, gene_subset=gsub), None
                elif cfg["model"] in ("gnn",):
                    return model(mb, eb, lb, gene_subset=gsub), None
                else:  # depplm
                    tk = torch.from_numpy(tokens[ci]).to(device)
                    tm = torch.from_numpy(token_mask[ci]).to(device)
                    out = model(mb, eb, lb, tk, tm, gene_subset=gsub)
                    return out[0], out[1]
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred, sel_pred = _fw()
            else:
                pred, sel_pred = _fw()
            m = torch.isfinite(yb)
            diff = torch.where(m, pred - yb, torch.zeros_like(pred))
            loss = (diff ** 2).sum() / m.sum().clamp(min=1)
            if sel_pred is not None:
                loss = loss + cfg["lamb_sel"] * F_bce(
                    sel_pred, torch.from_numpy(sel_y).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(ci)
        sched.step()
        val_loss, val_metrics = evaluate(model, cfg, Y, mut, expr, feat, lin_codes,
                                         tokens, token_mask, va_idx, sel_y, device,
                                         n_lin)
        if epoch % 5 == 0 or epoch == max_epochs - 1:
            print(f"[{run_tag}] ep {epoch}: train {tot/len(tr_idx):.4f} "
                  f"val {val_loss:.4f} pearson {val_metrics['pearson']:.4f}", flush=True)
        if val_loss < best_val - 1e-4:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[{run_tag}] early stop at epoch {epoch}", flush=True)
                break
        if epoch > 0 and abs(prev_val - val_loss) < 1e-6:
            conv += 1
            if conv >= 5:
                print(f"[{run_tag}] convergence gate at epoch {epoch}", flush=True)
                break
        else:
            conv = 0
        prev_val = val_loss
    if best_state is not None:
        model.load_state_dict(best_state)
    _, final_metrics = evaluate(model, cfg, Y, mut, expr, feat, lin_codes,
                                tokens, token_mask, va_idx, sel_y, device, n_lin,
                                final=True)
    return final_metrics


def F_mse(p, y):
    return torch.nn.functional.mse_loss(p, y)


def F_bce(p, y):
    return torch.nn.functional.binary_cross_entropy_with_logits(p, y)


def evaluate(model, cfg, Y, mut, expr, feat, lin_codes, tokens, token_mask,
             idx, sel_y, device, n_lin, final=False):
    model.eval()
    loss_sum, n_pairs, r_list = 0.0, 0, []
    mae_sum, sp_list = 0.0, []
    with torch.inference_mode():
        for b in range(0, len(idx), 64):
            ci = idx[b:b + 64]
            mb = torch.from_numpy(mut[ci]).to(device)
            eb = torch.from_numpy(expr[ci]).to(device)
            lb = torch.zeros((len(ci), n_lin), device=device).scatter_(
                1, torch.from_numpy(lin_codes[ci]).long().unsqueeze(1).to(device), 1.0)
            yb = torch.from_numpy(Y[:, ci].T).to(device)
            if cfg["model"] == "deepdep":
                pred = model(mb, eb, lb)
            elif cfg["model"] == "featmlp":
                pred = model(torch.from_numpy(feat).to(device), mb, eb, lb)
            elif cfg["model"] == "onehot":
                pred = model(mb, eb, lb)
            elif cfg["model"] == "plmmlp":
                pred = model(mb, eb, lb)
            elif cfg["model"] == "gnn":
                pred = model(mb, eb, lb)
            else:
                tk = torch.from_numpy(tokens[ci]).to(device)
                tm = torch.from_numpy(token_mask[ci]).to(device)
                pred, _, _ = model(mb, eb, lb, tk, tm)
            P_b = pred.cpu().numpy()
            T_b = yb.cpu().numpy()
            m_b = np.isfinite(T_b)
            diff = P_b - T_b
            loss_sum += float(np.sum(diff[m_b] ** 2))
            n_pairs += int(m_b.sum())
            mae_sum += float(np.sum(np.abs(diff[m_b])))
            for i in range(P_b.shape[0]):
                mi = m_b[i]
                if mi.sum() < 10:
                    r_list.append(np.nan)
                    continue
                r_list.append(np.corrcoef(P_b[i][mi], T_b[i][mi])[0, 1])
                if final:
                    from scipy.stats import rankdata
                    rx = rankdata(P_b[i][mi])
                    ry = rankdata(T_b[i][mi])
                    sp_list.append(np.corrcoef(rx, ry)[0, 1])
    r = np.array(r_list)
    loss = loss_sum / max(n_pairs, 1)
    metrics = {"loss": loss, "pearson": float(np.nanmean(r)),
               "pearson_sd": float(np.nanstd(r)),
               "mae": mae_sum / max(n_pairs, 1)}
    metrics["spearman"] = float(np.nanmean(sp_list)) if final and sp_list else np.nan
    return loss, metrics


def masked_pearson(P, T, m):
    """Per-cell-line Pearson on finite-label genes."""
    from scipy.stats import rankdata
    rs = []
    for i in range(P.shape[0]):
        mi = m[i]
        if mi.sum() < 10:
            rs.append(np.nan)
            continue
        rs.append(np.corrcoef(P[i][mi], T[i][mi])[0, 1])
    return np.array(rs)


def masked_spearman(P, T, m):
    """Per-cell-line Spearman on finite-label genes."""
    from scipy.stats import rankdata
    rs = []
    for i in range(P.shape[0]):
        mi = m[i]
        if mi.sum() < 10:
            rs.append(np.nan)
            continue
        rx = rankdata(P[i][mi])
        ry = rankdata(T[i][mi])
        rs.append(np.corrcoef(rx, ry)[0, 1])
    return np.array(rs)


# --------------------------------------------------------------- experiment
CONDS = {
    "C-DEEPDEP": {"model": "deepdep"},
    "C-FEAT": {"model": "featmlp"},
    "C-ONEHOT": {"model": "onehot"},
    "C-PLM-MLP": {"model": "plmmlp"},
    "C-GCN": {"model": "gnn", "conv": "gcn", "comut": False},
    "C-GAT": {"model": "gnn", "conv": "gat", "comut": False},
    "C-GCN-comut": {"model": "gnn", "conv": "gcn", "comut": True},
    "DepPLM": {"model": "depplm"},
    "A1-ESM": {"model": "depplm", "use_pretrained": False},
    "A2-GNN": {"model": "depplm", "use_gnn": False},
    "A3-comut": {"model": "depplm", "use_comut": False},
    "A4-MT": {"model": "depplm", "use_multitask": False},
}

GRID = [{"hidden": h, "lr": lr, "wd": wd}
        for h in (128, 256) for lr in (1e-3, 3e-4) for wd in (1e-5,)]
# NOTE (2026-10-18 scaling decision): wd=1e-4 dropped from the shared grid to
# keep the equal-budget tuning tractable on a single GPU; logged in PROGRESS.md


def build_model(cfg, emb, adj, edge_index, edge_weight, n_mut, n_expr, n_lin, G):
    h = cfg.get("hidden", 128)
    if cfg["model"] == "deepdep":
        return DeepDepMLP(G, n_mut, n_expr, n_lin, h)
    if cfg["model"] == "featmlp":
        return FeatureMLP(4, n_mut, n_expr, n_lin, G, h)
    if cfg["model"] == "onehot":
        return OneHotMLP(G, n_mut, n_expr, n_lin, h)
    if cfg["model"] == "plmmlp":
        return PLMMLP(torch.from_numpy(emb), n_mut, n_expr, n_lin, h)
    if cfg["model"] == "gnn":
        ew = edge_weight if cfg.get("comut") else None
        return GNNRegressor(torch.from_numpy(emb), adj, edge_index, ew,
                            n_mut, n_expr, n_lin, cfg.get("conv", "gcn"), h)
    return DepPLM(torch.from_numpy(emb), adj, edge_index, edge_weight,
                  n_mut, n_expr, n_lin, h, k_tokens=32,
                  use_pretrained=cfg.get("use_pretrained", True),
                  use_gnn=cfg.get("use_gnn", True),
                  use_comut=cfg.get("use_comut", True),
                  use_multitask=cfg.get("use_multitask", True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--cond", default=None, help="comma-separated condition ids")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--time-budget", type=float, default=5 * 3600)
    ap.add_argument("--fold-id", type=int, default=-1, help="run only this fold (default: all)")
    args = ap.parse_args()
    global TIME_BUDGET, T0
    TIME_BUDGET, T0 = args.time_budget, time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[06] device={device} pilot={args.pilot}", flush=True)

    (Y, mut, expr, feat, lin_codes, emb, edge_index, edge_weight, adj,
     pilot_ci, pilot_gi) = load_data(pilot=args.pilot)
    tokens, token_mask = build_tokens(mut, gi=pilot_gi)
    sel_y, mad = selectivity_labels(Y)
    n_lin = int(np.max(lin_codes)) + 1
    print(f"[06] Y {Y.shape} mut {mut.shape} expr {expr.shape} emb {emb.shape} "
          f"edges {edge_index.shape[1]}", flush=True)

    # splits (P1 5-fold over cell lines, fixed seed)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(Y.shape[1])
    folds = np.array_split(perm, args.folds)
    splits_json = json.loads((DATA / "splits.json").read_text())
    lin_codes = np.array(lin_codes)

    cond_ids = args.cond.split(",") if args.cond else list(CONDS)
    for cond_id in cond_ids:
        cfg = dict(CONDS[cond_id])
        # equal-budget tuning for every condition (baselines too)
        best_cfg, best_score = None, -1e9
        for g in GRID:
            c2 = {**cfg, **g}
            c2["lamb_sel"] = 0.1
            if args.pilot:
                c2["lamb_sel"] = 0.1
                m = build_model(c2, emb, adj, edge_index, edge_weight,
                                mut.shape[1], expr.shape[1], n_lin, Y.shape[0])
                t0 = time.time()
                tr, va = perm[:160], perm[160:]
                train_eval(m, c2, Y, mut, expr, feat, lin_codes, emb,
                           edge_index, edge_weight, adj, tokens, token_mask,
                           tr, va, sel_y, device, f"PILOT-{cond_id}", pilot=True)
                dt = time.time() - t0
                n_grid = len(GRID)
                est = dt * 30 * 5 * n_grid / 60  # rough: epochs*folds*grid
                print(f"TIME_ESTIMATE: pilot 1 epoch = {dt:.1f}s -> "
                      f"full condition ~ {est:.0f} min", flush=True)
                continue
            # quick fold-0 selection over grid (train 40%, val 10%)
            tr, va = folds[0][: int(0.8 * len(folds[0]))], folds[0][int(0.8 * len(folds[0])):]
            m = build_model(c2, emb, adj, edge_index, edge_weight,
                            mut.shape[1], expr.shape[1], n_lin, Y.shape[0])
            met = train_eval(m, c2, Y, mut, expr, feat, lin_codes, emb,
                             edge_index, edge_weight, adj, tokens, token_mask,
                             tr, va, sel_y, device, f"{cond_id}-tune",
                             max_epochs=15, patience=6)
            print(f"[06] {cond_id} grid {g}: val pearson {met['pearson']:.4f}", flush=True)
            with (RES / "tuning_log.jsonl").open("a") as fh:
                fh.write(json.dumps({"cond": cond_id, "grid": g,
                                     "val_pearson": met["pearson"],
                                     "wall_s": time.time() - T0}) + "\n")
            if met["pearson"] > best_score:
                best_score, best_cfg = met["pearson"], c2
            if time_guard():
                break
        if args.pilot:
            continue
        print(f"[06] {cond_id} BEST grid: { {k: best_cfg[k] for k in GRID[0]} } "
              f"val pearson {best_score:.4f}", flush=True)
        # full 5-fold CV with best cfg
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        fold_metrics, per_fold = [], []
        for fi in range(args.folds):
            if args.fold_id >= 0 and fi != args.fold_id:
                continue
            va = folds[fi]
            tr = np.concatenate([folds[j] for j in range(args.folds) if j != fi])
            torch.manual_seed(args.seed + fi)
            np.random.seed(args.seed + fi)
            m = build_model(best_cfg, emb, adj, edge_index, edge_weight,
                            mut.shape[1], expr.shape[1], n_lin, Y.shape[0])
            met = train_eval(m, best_cfg, Y, mut, expr, feat, lin_codes, emb,
                             edge_index, edge_weight, adj, tokens, token_mask,
                             tr, va, sel_y, device, f"{cond_id}-f{fi}",
                             max_epochs=30, patience=8)
            fold_metrics.append(met)
            per_fold.append({"fold": fi, "metrics": met})
            print(f"[06] {cond_id} fold {fi}: pearson {met['pearson']:.4f}", flush=True)
            if time_guard():
                break
        if fold_metrics:
            summary = {"cond": cond_id, "best_grid": {k: best_cfg[k] for k in GRID[0]},
                       "seed": args.seed,
                       "mean": {k: float(np.mean([f[k] for f in fold_metrics]))
                                for k in ("pearson", "spearman", "mae", "loss")},
                       "sd": {k: float(np.std([f[k] for f in fold_metrics]))
                              for k in ("pearson", "spearman", "mae", "loss")},
                       "folds": per_fold,
                       "wall_s": time.time() - T0}
            (RES / f"cond_{cond_id}.json").write_text(json.dumps(summary, indent=1),
                                                      encoding="utf-8")
            print(f"[06] {cond_id} DONE: {summary['mean']}", flush=True)


if __name__ == "__main__":
    main()
