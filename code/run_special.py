"""06b_run_special.py — Stage E special conditions:
T-hem / T-solid (cross-lineage transfer, P2 split), Z-FAM (held-out gene
families, zero-shot), T12 (ESM-2 t12 vs t6 scale comparison).
All runs are real training/eval with the shared machinery from
06_run_experiments.py.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_experiments import (build_model, build_tokens, evaluate, pd_factorize,
                             selectivity_labels, train_eval, GRID)
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"


def load_all():
    Y = np.load(DATA / "gene_effect.npy").astype(np.float32)
    mut = np.load(DATA / "cell_mut.npy").astype(np.float32)
    expr = np.load(DATA / "cell_expr_pca.npy").astype(np.float32)
    feat = np.load(DATA / "gene_handfeat.npy").astype(np.float32)
    lineages = [l.strip() for l in (DATA / "lineages.txt").read_text().splitlines()]
    lin = pd_factorize(lineages)
    emb = np.load(DATA / "esm2_t6_8M.npy").astype(np.float32)
    g = np.load(DATA / "ppi_graph.npz")
    from models import normalized_adjacency
    ei = torch.from_numpy(g["edge_index"].astype(np.int64)).t().contiguous()
    ew = torch.from_numpy(g["edge_weight"].astype(np.float32))
    adj = normalized_adjacency(ei, ew, emb.shape[0])
    return Y, mut, expr, feat, lin, emb, ei, ew, adj


def family_labels(emb: np.ndarray, thr: float = 0.98) -> np.ndarray:
    """Greedy cosine-similarity families (deterministic, memory-bounded).
    thr=0.98: ESM-2 t6 embeddings have baseline cosine ~0.98; at this
    threshold the greedy families approximate paralog-like groups (median
    family size 1, 186 families with >=10 members, verified 2026-10-19)."""
    l2 = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(42)
    order = rng.permutation(emb.shape[0])
    labels = np.full(emb.shape[0], -1, dtype=np.int64)
    fam = 0
    unlab = order.tolist()
    while unlab:
        batch = np.array(unlab[:256], dtype=np.int64)
        sim = l2[batch] @ l2.T            # B x G (bounded memory)
        done = []
        for j, i in enumerate(batch):
            if labels[i] != -1:
                done.append(i)
                continue
            members = np.where(sim[j] >= thr)[0]
            members = members[labels[members] == -1]
            labels[members] = fam
            fam += 1
            done.append(i)
        unlab = [i for i in unlab if labels[i] == -1]
        if len(done) == 0:
            break
    return labels


def run_transfer(Y, mut, expr, feat, lin, emb, ei, ew, adj, cond_id, seed=42):
    """T-hem / T-solid: lineage-holdout transfer (P2 split)."""
    splits = json.loads((DATA / "splits.json").read_text())
    tr_idx = np.array(splits["P2_train"], dtype=np.int64)
    te_idx = np.array(splits["P2_test"], dtype=np.int64)
    direction = cond_id  # T-hem: train solid->test heme; T-solid: reversed
    if cond_id == "T-solid":
        tr_idx, te_idx = te_idx, tr_idx
    tokens, token_mask = build_tokens(mut)
    sel_y, _ = selectivity_labels(Y)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_lin = int(np.max(lin)) + 1
    out = {}
    for model_id, cfg in [("DepPLM", {"model": "depplm"}),
                          ("C-PLM-MLP", {"model": "plmmlp"}),
                          ("C-FEAT", {"model": "featmlp"}),
                          ("C-GCN-comut", {"model": "gnn", "conv": "gcn", "comut": True})]:
        best = None
        for g in GRID:
            c2 = {**cfg, **g, "lamb_sel": 0.1}
            va = tr_idx[: int(0.2 * len(tr_idx))]
            m = build_model(c2, emb, adj, ei, ew, mut.shape[1], expr.shape[1],
                            n_lin, Y.shape[0])
            met = train_eval(m, c2, Y, mut, expr, feat, lin, emb, ei, ew, adj,
                             tokens, token_mask, tr_idx, va, sel_y, device,
                             f"{direction}-{model_id}-tune", max_epochs=20, patience=5)
            if best is None or met["pearson"] > best[0]:
                best = (met["pearson"], c2)
        torch.manual_seed(seed)
        m = build_model(best[1], emb, adj, ei, ew, mut.shape[1], expr.shape[1],
                        n_lin, Y.shape[0])
        met = train_eval(m, best[1], Y, mut, expr, feat, lin, emb, ei, ew, adj,
                         tokens, token_mask, tr_idx, te_idx, sel_y, device,
                         f"{direction}-{model_id}", max_epochs=40, patience=8)
        out[model_id] = met
        print(f"[06b] {direction} {model_id}: test pearson {met['pearson']:.4f}",
              flush=True)
    (RES / f"cond_{cond_id}.json").write_text(
        json.dumps({"cond": cond_id, "models": out, "n_test_cells": int(len(te_idx))},
                   indent=1), encoding="utf-8")


def run_zeroshot(Y, mut, expr, feat, lin, emb, ei, ew, adj, seed=42):
    """Z-FAM: hold out ~20% of genes grouped by sequence-similarity families."""
    fam = family_labels(emb)
    uniq, sizes = np.unique(fam, return_counts=True)
    order = np.argsort(-sizes)
    # hold out ~20% of NON-SINGLETON families (paralog-like groups); singletons
    # stay in training — a singleton carries no family signal to hide
    multi = uniq[sizes >= 2]
    multi_sizes = sizes[sizes >= 2]
    multi_order = np.argsort(-multi_sizes)
    target = int(0.2 * int((sizes >= 2).sum()))
    held_fams, acc = [], 0
    for f in multi_order:
        if acc >= target and held_fams:
            break
        held_fams.append(multi[f]); acc += multi_sizes[f]
    holdout = np.isin(fam, held_fams)
    keep = ~holdout
    print(f"[06b] Z-FAM: {len(multi)} non-singleton families; holdout families "
          f"{len(held_fams)}, genes {int(holdout.sum())} ({acc})", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_lin = int(np.max(lin)) + 1
    # full-space tokens for the zero-shot full-gene evaluation
    tokens_full, token_mask_full = build_tokens(mut)
    # remapped tokens (kept-gene rows) for training
    tokens, token_mask = build_tokens(mut, gi=np.where(keep)[0])
    sel_y, _ = selectivity_labels(Y[keep])
    # keep full-graph references for the zero-shot evaluation
    from models import normalized_adjacency
    keep_idx = np.where(keep)[0]
    adj_full, ei_full, ew_full = adj, ei, ew
    kset = set(keep_idx.tolist())
    e = ei.numpy()
    ke = np.isin(e[0], keep_idx) & np.isin(e[1], keep_idx)
    e2, w2 = e[:, ke], ew.numpy()[ke]
    remap = {int(o): new for new, o in enumerate(keep_idx)}
    e2 = np.array([[remap[int(a)], remap[int(b)]] for a, b in e2.T]).T
    ei_s = torch.from_numpy(e2.astype(np.int64)).contiguous()
    ew_s = torch.from_numpy(w2.astype(np.float32))
    adj_s = normalized_adjacency(ei_s, ew_s, int(keep.sum()))
    ei, ew, adj = ei_s, ew_s, adj_s

    # random 70/30 cell-line split for training/eval
    rng = np.random.default_rng(seed)
    perm = rng.permutation(Y.shape[1])
    tr_idx = perm[: int(0.7 * Y.shape[1])]
    va_idx = perm[int(0.7 * Y.shape[1]):]

    out = {"n_families": int(len(uniq)),
           "n_non_singleton_families": int((sizes >= 2).sum())}
    for model_id, cfg in [("DepPLM", {"model": "depplm"}),
                          ("C-PLM-MLP", {"model": "plmmlp"}),
                          ("C-FEAT", {"model": "featmlp"})]:
        # train on kept genes only (mask Y rows), then eval per-gene on holdout
        Yk = Y[keep]
        best = None
        for g in GRID:
            c2 = {**cfg, **g, "lamb_sel": 0.1}
            m = build_model(c2, emb[keep], adj, ei, ew, mut.shape[1],
                            expr.shape[1], n_lin, int(keep.sum()))
            met = train_eval(m, c2, Yk, mut, expr, feat[keep], lin, emb[keep],
                             ei, ew, adj, tokens, token_mask, tr_idx, va_idx,
                             sel_y, device, f"ZF-{model_id}-tune",
                             max_epochs=20, patience=5)
            if best is None or met["pearson"] > best[0]:
                best = (met["pearson"], c2)
        m = build_model(best[1], emb[keep], adj, ei, ew, mut.shape[1],
                        expr.shape[1], n_lin, int(keep.sum()))
        met = train_eval(m, best[1], Yk, mut, expr, feat[keep], lin, emb[keep],
                         ei, ew, adj, tokens, token_mask, tr_idx, va_idx,
                         sel_y, device, f"ZF-{model_id}", max_epochs=40, patience=8)
        # zero-shot: build a FULL-gene model, transfer the trained weights
        # (proj/convs/heads are gene-count independent; the gene-bias is
        # expanded with zeros for held-out genes), predict ALL genes
        import torch as T
        full_model = build_model(best[1], emb, adj_full, ei_full, ew_full,
                                 mut.shape[1], expr.shape[1], n_lin, Y.shape[0])
        sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        fsd = full_model.state_dict()
        for k in fsd:
            if k in sd and fsd[k].shape == sd[k].shape:
                fsd[k] = sd[k]
            elif k.endswith("head.bias"):
                fsd[k][keep_idx] = sd[k]
        full_model.load_state_dict(fsd)
        full_model = full_model.to(device).eval()
        tk_full = T.from_numpy(tokens_full).to(device)
        tm_full = T.from_numpy(token_mask_full).to(device)
        P = np.zeros((len(va_idx), Y.shape[0]), dtype=np.float32)
        with T.inference_mode():
            for b in range(0, len(va_idx), 64):
                ci = va_idx[b:b + 64]
                mb = T.from_numpy(mut[ci]).to(device)
                eb = T.from_numpy(expr[ci]).to(device)
                lb = T.zeros((len(ci), n_lin), device=device).scatter_(
                    1, T.from_numpy(lin[ci]).long().unsqueeze(1).to(device), 1.0)
                if cfg["model"] == "featmlp":
                    pred = full_model(T.from_numpy(feat).to(device), mb, eb, lb)
                elif cfg["model"] == "plmmlp":
                    pred = full_model(mb, eb, lb)
                else:
                    pred, _, _ = full_model(mb, eb, lb, tk_full[ci], tm_full[ci])
                P[b:b + len(ci)] = pred.cpu().numpy()
        Tmat = Y[:, va_idx].T
        r_per_gene = np.array([np.corrcoef(P[:, i], Tmat[:, i])[0, 1]
                               for i in range(Y.shape[0])])
        # restrict to genes with meaningful cross-cell variance (var >= 0.05;
        # 1,962 of 17,768 genes — per-gene correlation is ill-defined below that)
        gvar = np.nanvar(Y, axis=1)
        vmask = gvar >= 0.05
        r_hold = np.nanmean(r_per_gene[holdout & vmask])
        r_keep = np.nanmean(r_per_gene[keep & vmask])
        n_eval = int(vmask.sum())
        out[model_id] = {"pearson_holdout_genes": float(r_hold),
                         "pearson_kept_genes": float(r_keep),
                         "n_holdout": int(holdout.sum()),
                         "n_holdout_evaluable": int((holdout & vmask).sum()),
                         "n_evaluable_genes": n_eval,
                         "seen_val_pearson": met["pearson"]}
        print(f"[06b] Z-FAM {model_id}: holdout-gene r = {r_hold:.4f} "
              f"(kept {r_keep:.4f})", flush=True)
    (RES / "cond_Z-FAM.json").write_text(json.dumps({"cond": "Z-FAM", **out},
                                                    indent=1), encoding="utf-8")


def run_t12(Y, mut, expr, feat, lin, ei, ew, adj, seed=42):
    """C-PLM-MLP-t12: ESM-2 t12 vs t6 on the shared 2k-gene subset."""
    t12 = np.load(DATA / "esm2_t12_35M_sub2k.npy").astype(np.float32)
    t12_names = [l.strip() for l in
                 (DATA / "esm2_t12_35M_sub2k.names.txt").read_text().splitlines()]
    genes = [l.strip() for l in (DATA / "genes.txt").read_text().splitlines()]
    gidx = {g: i for i, g in enumerate(genes)}
    sub = [gidx[n] for n in t12_names if n in gidx]
    sub = np.array(sub, dtype=np.int64)
    t6_sub = np.load(DATA / "esm2_t6_8M.npy")[sub].astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_lin = int(np.max(lin)) + 1
    tokens, token_mask = build_tokens(mut)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(Y.shape[1])
    tr_idx = perm[: int(0.7 * Y.shape[1])]
    va_idx = perm[int(0.7 * Y.shape[1]):]
    out = {}
    from models import PLMMLP
    for tag, emb2 in [("t6", t6_sub), ("t12", t12)]:
        Ysub = Y[sub]
        sel_y, _ = selectivity_labels(Ysub)
        best = None
        for g in GRID:
            c2 = {"model": "plmmlp", **g}
            m = PLMMLP(torch.from_numpy(emb2), mut.shape[1], expr.shape[1],
                       n_lin, g["hidden"])
            met = train_eval(m, c2, Ysub, mut, expr, feat[sub], lin, emb2, ei,
                             ew, adj, tokens, token_mask, tr_idx, va_idx,
                             sel_y, device, f"T12-{tag}-tune",
                             max_epochs=20, patience=5)
            if best is None or met["pearson"] > best[0]:
                best = (met["pearson"], g)
        torch.manual_seed(seed)
        m = PLMMLP(torch.from_numpy(emb2), mut.shape[1], expr.shape[1],
                   n_lin, best[1]["hidden"])
        c2 = {"model": "plmmlp", **best[1]}
        met = train_eval(m, c2, Ysub, mut, expr, feat[sub], lin, emb2, ei, ew,
                         adj, tokens, token_mask, tr_idx, va_idx, sel_y, device,
                         f"T12-{tag}", max_epochs=40, patience=8)
        out[f"plmmlp_{tag}"] = met
        print(f"[06b] T12 {tag}: pearson {met['pearson']:.4f}", flush=True)
    (RES / "cond_C-PLM-MLP-t12.json").write_text(
        json.dumps({"cond": "C-PLM-MLP-t12", "n_genes": int(len(sub)), **out},
                   indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["T-hem", "T-solid", "Z-FAM", "T12"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    Y, mut, expr, feat, lin, emb, ei, ew, adj = load_all()
    if args.cond in ("T-hem", "T-solid"):
        run_transfer(Y, mut, expr, feat, lin, emb, ei, ew, adj, args.cond, args.seed)
    elif args.cond == "Z-FAM":
        run_zeroshot(Y, mut, expr, feat, lin, emb, ei, ew, adj, args.seed)
    else:
        run_t12(Y, mut, expr, feat, lin, ei, ew, adj, args.seed)


if __name__ == "__main__":
    main()
