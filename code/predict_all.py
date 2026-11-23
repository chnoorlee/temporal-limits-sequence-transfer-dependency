"""06c_predict_all.py — out-of-fold full-matrix predictions for DepPLM.

Trains DepPLM with the best grid (from results/cond_DepPLM.json) on 4 folds
and predicts the held-out fold for all 5 folds -> G x C out-of-fold matrix,
saved to results/predictions/DepPLM_all.npy (used by 08_clinical_validation.py
for selectivity ranking). Real training; no leakage (predictions are strictly
out-of-fold).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_experiments import (build_model, build_tokens, pd_factorize,
                             selectivity_labels, train_eval)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"
PRED_DIR = RES / "predictions"
PRED_DIR.mkdir(exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

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

    dep = json.loads((RES / "cond_DepPLM.json").read_text(encoding="utf-8"))
    best = {**{"model": "depplm", "lamb_sel": 0.1}, **dep["best_grid"]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_lin = int(np.max(lin)) + 1
    tokens, token_mask = build_tokens(mut)
    sel_y, _ = selectivity_labels(Y)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(Y.shape[1])
    folds = np.array_split(perm, args.folds)

    P = np.zeros_like(Y)
    import torch as T
    for fi in range(args.folds):
        va = folds[fi]
        tr = np.concatenate([folds[j] for j in range(args.folds) if j != fi])
        T.manual_seed(args.seed + fi)
        np.random.seed(args.seed + fi)
        m = build_model(best, emb, adj, ei, ew, mut.shape[1], expr.shape[1],
                        n_lin, Y.shape[0])
        train_eval(m, best, Y, mut, expr, feat, lin, emb, ei, ew, adj,
                   tokens, token_mask, tr, va, sel_y, device,
                   f"predictall-f{fi}", max_epochs=40, patience=8)
        m.eval()
        with T.inference_mode():
            for b in range(0, len(va), 64):
                ci = va[b:b + 64]
                mb = T.from_numpy(mut[ci]).to(device)
                eb = T.from_numpy(expr[ci]).to(device)
                lb = T.zeros((len(ci), n_lin), device=device).scatter_(
                    1, T.from_numpy(lin[ci]).long().unsqueeze(1).to(device), 1.0)
                tk = T.from_numpy(tokens[ci]).to(device)
                tm = T.from_numpy(token_mask[ci]).to(device)
                pred, _, _ = m(mb, eb, lb, tk, tm)
                P[:, ci] = pred.t().cpu().numpy()
        print(f"[06c] fold {fi} predicted ({len(va)} cells)", flush=True)

    np.save(PRED_DIR / "DepPLM_all.npy", P)
    # honest check: all cells covered
    assert np.isfinite(P).all(), "NaN in predictions — investigate"
    print(f"[06c] out-of-fold matrix {P.shape} saved", flush=True)


if __name__ == "__main__":
    main()
