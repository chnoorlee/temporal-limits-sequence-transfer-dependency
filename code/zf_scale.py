"""zf_scale.py — optimization 1: does embedding capacity change the zero-shot
ceiling? Z-FAM protocol with ESM-2 t33-650M embeddings vs t6-8M on the SAME
family split, PLM-MLP model (the purest readout of the prior).

Writes results/cond_Z-FAM-t33.json (real numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import (build_model, build_tokens, pd_factorize,
                             selectivity_labels, train_eval, GRID)
from run_special import family_labels

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"


def load_all(emb_path: Path, emb_names_path: Path):
    Y = np.load(DATA / "gene_effect.npy").astype(np.float32)
    mut = np.load(DATA / "cell_mut.npy").astype(np.float32)
    expr = np.load(DATA / "cell_expr_pca.npy").astype(np.float32)
    feat = np.load(DATA / "gene_handfeat.npy").astype(np.float32)
    lineages = [l.strip() for l in (DATA / "lineages.txt").read_text().splitlines()]
    lin = pd_factorize(lineages)
    # embeddings in gene order
    emb_full = np.load(emb_path).astype(np.float32)
    names = [l.strip() for l in emb_names_path.read_text().splitlines()]
    genes = [l.strip() for l in (DATA / "genes.txt").read_text().splitlines()]
    gi = {g: i for i, g in enumerate(genes)}
    emb = np.zeros((len(genes), emb_full.shape[1]), dtype=np.float32)
    mapped = 0
    for i, n in enumerate(names):
        if n in gi:
            emb[gi[n]] = emb_full[i]
            mapped += 1
    print(f"[zfscale] mapped {mapped}/{len(names)} embeddings to gene order",
          flush=True)
    from models import normalized_adjacency
    g = np.load(DATA / "ppi_graph.npz")
    ei = torch.from_numpy(g["edge_index"].astype(np.int64)).t().contiguous()
    ew = torch.from_numpy(g["edge_weight"].astype(np.float32))
    adj = normalized_adjacency(ei, ew, len(genes))
    return Y, mut, expr, feat, lin, emb, ei, ew, adj


def run(emb, Y, mut, expr, feat, lin, ei, ew, adj, tag, device, n_lin,
        tokens, token_mask, holdout, keep, tr_idx, va_idx):
    fam = family_labels(emb)
    uniq, sizes = np.unique(fam, return_counts=True)
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
    print(f"[zfscale:{tag}] non-singleton families {(sizes >= 2).sum()}, "
          f"holdout genes {int(holdout.sum())}", flush=True)
    sel_y, _ = selectivity_labels(Y[keep])
    best = None
    for g in GRID:
        c2 = {"model": "plmmlp", **g}
        m = build_model(c2, emb[keep], adj, ei, ew, mut.shape[1], expr.shape[1],
                        n_lin, int(keep.sum()))
        met = train_eval(m, c2, Y[keep], mut, expr, feat[keep], lin, emb[keep],
                         ei, ew, adj, tokens, token_mask, tr_idx, va_idx,
                         sel_y, device, f"ZFSCALE-{tag}-tune",
                         max_epochs=20, patience=5)
        if best is None or met["pearson"] > best[0]:
            best = (met["pearson"], c2)
    m = build_model(best[1], emb[keep], adj, ei, ew, mut.shape[1], expr.shape[1],
                    n_lin, int(keep.sum()))
    met = train_eval(m, best[1], Y[keep], mut, expr, feat[keep], lin, emb[keep],
                     ei, ew, adj, tokens, token_mask, tr_idx, va_idx,
                     sel_y, device, f"ZFSCALE-{tag}", max_epochs=40, patience=8)
    # zero-shot on the FULL gene set: weight transfer to a full-gene model
    full_model = build_model(best[1], emb, adj, ei, ew, mut.shape[1],
                             expr.shape[1], n_lin, Y.shape[0])
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    fsd = full_model.state_dict()
    keep_idx = np.where(keep)[0]
    for k in fsd:
        if k in sd and fsd[k].shape == sd[k].shape:
            fsd[k] = sd[k]
        elif k.endswith("head.bias"):
            fsd[k][keep_idx] = sd[k]
    full_model.load_state_dict(fsd)
    full_model = full_model.to(device).eval()
    P = np.zeros((len(va_idx), Y.shape[0]), dtype=np.float32)
    with torch.inference_mode():
        for b in range(0, len(va_idx), 64):
            ci = va_idx[b:b + 64]
            mb = torch.from_numpy(mut[ci]).to(device)
            eb = torch.from_numpy(expr[ci]).to(device)
            lb = torch.zeros((len(ci), n_lin), device=device).scatter_(
                1, torch.from_numpy(lin[ci]).long().unsqueeze(1).to(device), 1.0)
            pred = full_model(mb, eb, lb)
            P[b:b + len(ci)] = pred.cpu().numpy()
    Tmat = Y[:, va_idx].T
    r_per_gene = np.array([np.corrcoef(P[:, i], Tmat[:, i])[0, 1]
                           for i in range(Y.shape[0])])
    gvar = np.nanvar(Y, axis=1)
    vmask = gvar >= 0.05
    r_hold = float(np.nanmean(r_per_gene[holdout & vmask]))
    r_keep = float(np.nanmean(r_per_gene[keep & vmask]))
    print(f"[zfscale:{tag}] holdout per-gene r = {r_hold:.4f} "
          f"(kept {r_keep:.4f}, seen-val {met['pearson']:.4f})", flush=True)
    return {"pearson_holdout_genes": r_hold, "pearson_kept_genes": r_keep,
            "n_holdout": int(holdout.sum()),
            "n_non_singleton_families": int((sizes >= 2).sum()),
            "seen_val_pearson": met["pearson"]}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(42)
    Y, mut, expr, feat, lin, _, ei, ew, adj = load_all(
        DATA / "esm2_t6_8M.npy", DATA / "esm2_t6_8M.names.txt")
    n_lin = int(np.max(lin)) + 1
    tokens, token_mask = build_tokens(mut)
    perm = rng.permutation(Y.shape[1])
    tr_idx = perm[: int(0.7 * Y.shape[1])]
    va_idx = perm[int(0.7 * Y.shape[1]):]
    out = {"cond": "Z-FAM-t33", "cell_split_seed": 42}
    for tag, emb_p, names_p in [("t6", DATA / "esm2_t6_8M.npy",
                                 DATA / "esm2_t6_8M.names.txt"),
                                ("t33", DATA / "esm2_t33_650M.npy",
                                 DATA / "esm2_t33_650M.names.txt")]:
        if not emb_p.exists():
            print(f"[zfscale] {emb_p.name} missing — skipping", flush=True)
            continue
        Y2, mut2, expr2, feat2, lin2, emb2, ei2, ew2, adj2 = load_all(emb_p, names_p)
        out[tag] = run(emb2, Y2, mut2, expr2, feat2, lin2, ei2, ew2, adj2,
                       tag, device, n_lin, tokens, token_mask, None, None,
                       tr_idx, va_idx)
    (RES / "cond_Z-FAM-t33.json").write_text(json.dumps(out, indent=1),
                                             encoding="utf-8")
    print("[zfscale] cond_Z-FAM-t33.json written", flush=True)


if __name__ == "__main__":
    main()
