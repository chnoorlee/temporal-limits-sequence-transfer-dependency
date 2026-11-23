"""case_study.py — Stage F: candidate-target case study from real results.

1. Select the top candidate: top-500 predicted-selective ∩ SM/AB-tractable
   ∩ no Phase-II+ clinical drug.
2. Compute its lineage-selective dependency pattern (mean predicted effect
   per OncoTree lineage, out-of-fold predictions).
3. Extract cross-attention partners: retrain DepPLM (best grid) on 80% of
   cell lines and average attention over the candidate gene's top partners
   across cell lines where the candidate is predicted most essential.
4. STRING-neighborhood enrichment of the partners among observed pan-cancer
   essential genes vs random expectation.
5. Screening coverage (fraction of cell lines with measured label).
Writes results/cond_CASE.json (all real numbers).
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"


def main() -> None:
    genes = [l.strip() for l in (DATA / "genes.txt").read_text().splitlines()]
    P = np.load(RES / "predictions" / "DepPLM_all.npy")  # G x C
    Y = np.load(DATA / "gene_effect.npy")
    clin = json.loads((RES / "cond_CLIN.json").read_text(encoding="utf-8"))
    top500 = [g.upper() for g in clin["top500_genes"]]
    phase2 = set(g.upper() for g in clin.get("phase2plus_top500", []))
    druggable = set(g.upper() for g in clin.get("tractable_top500", []))

    sel = 1.4826 * np.nanmedian(np.abs(P - np.nanmedian(P, axis=1, keepdims=True)),
                                axis=1)
    rank = {genes[i].upper(): i for i in range(len(genes))}
    candidates = [g for g in top500 if g in druggable and g not in phase2]
    print(f"[case] candidates (top500 ∩ druggable ∩ no PhaseII+): {len(candidates)}",
          flush=True)
    if not candidates:
        print("[case] no candidates — check CLIN data", flush=True)
        return
    gene = candidates[0]
    gi = rank[gene]

    # lineage pattern
    lineages = [l.strip() for l in (DATA / "lineages.txt").read_text().splitlines()]
    uni = sorted(set(lineages))
    lin_mean = {}
    for li in uni:
        cells = [i for i, l in enumerate(lineages) if l == li]
        lin_mean[li] = float(np.nanmean(P[gi, cells]))
    top_lin = sorted(lin_mean.items(), key=lambda kv: kv[1])[:4]

    # screening coverage
    cov = float(np.isfinite(Y[gi]).mean())

    # attention partners: quick retrain of DepPLM (best grid) on 80% cells
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from run_experiments import (build_model, build_tokens, load_data,
                                 selectivity_labels, train_eval)
    from models import DepPLM
    Yd, mut, expr, feat, lin, emb, ei, ew, adj, _, _ = load_data(pilot=False)
    dep_cfg = json.loads((RES / "cond_DepPLM.json").read_text(encoding="utf-8"))
    best = {**{"model": "depplm", "lamb_sel": 0.1}, **dep_cfg["best_grid"]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_lin = int(np.max(lin)) + 1
    tokens, token_mask = build_tokens(mut)
    sel_y, _ = selectivity_labels(Yd)
    rng = np.random.default_rng(42)
    perm = rng.permutation(Yd.shape[1])
    tr = perm[: int(0.8 * Yd.shape[1])]
    va = perm[int(0.8 * Yd.shape[1]):]
    m = build_model(best, emb, adj, ei, ew, mut.shape[1], expr.shape[1],
                    n_lin, Yd.shape[0])
    train_eval(m, best, Yd, mut, expr, feat, lin, emb, ei, ew, adj,
               tokens, token_mask, tr, va, sel_y, device, "case-study",
               max_epochs=30, patience=8)
    m.eval()
    # cells where the gene is predicted most essential (bottom 10% of predictions)
    pred_g = P[gi]
    focus = np.argsort(pred_g)[: max(int(0.1 * len(pred_g)), 20)]
    attn_sum = np.zeros(len(genes))
    with torch.inference_mode():
        for b in range(0, len(focus), 32):
            ci = focus[b:b + 32]
            mb = torch.from_numpy(mut[ci]).to(device)
            eb = torch.from_numpy(expr[ci]).to(device)
            lb = torch.zeros((len(ci), n_lin), device=device).scatter_(
                1, torch.from_numpy(lin[ci]).long().unsqueeze(1).to(device), 1.0)
            tk = torch.from_numpy(tokens[ci]).to(device)
            tm = torch.from_numpy(token_mask[ci]).to(device)
            _, _, attn = m(mb, eb, lb, tk, tm)
            a = attn[:, gi].cpu().numpy()          # B,K
            for j in range(a.shape[0]):
                for k in range(a.shape[1]):
                    if tm[j, k]:
                        attn_sum[tk[j, k].item()] += a[j, k]
    top_idx = [i for i in np.argsort(-attn_sum)[:5] if attn_sum[i] > 0]
    top_partners = [genes[i] for i in top_idx]
    top_weights = [round(float(attn_sum[i]), 4) for i in top_idx]

    # STRING neighborhood enrichment of top-25 partners among observed
    # pan-essential genes (mean gene effect < -1 across >= 90% cell lines)
    g = np.load(DATA / "ppi_graph.npz")
    ei = g["edge_index"]
    neighbors = set()
    for a, b in ei:
        if a == gi:
            neighbors.add(int(b))
        elif b == gi:
            neighbors.add(int(a))
    paness = np.nanmean(Y < -1, axis=1) >= 0.9
    n_nb = len(neighbors)
    n_nb_ess = int(sum(paness[i] for i in neighbors if i < len(paness)))
    bg_rate = float(paness.mean())
    expected = n_nb * bg_rate
    from scipy.stats import hypergeom
    n_ess_total = int(paness.sum())
    p_hyper = float(hypergeom.sf(n_nb_ess - 1, len(genes), n_ess_total, n_nb)) \
        if n_nb > 0 else np.nan

    out = {
        "cond": "CASE",
        "gene": gene,
        "n_candidates": len(candidates),
        "candidate_list": candidates,
        "top_lineages": [(l, round(v, 3)) for l, v in top_lin],
        "n_screened_cells": int(np.isfinite(Y[gi]).sum()),
        "coverage": round(cov, 4),
        "top_attention_partners": top_partners,
        "top_attention_weights": top_weights,
        "n_string_neighbors": n_nb,
        "n_neighbors_pan_essential": n_nb_ess,
        "expected_by_chance": round(expected, 1),
        "hypergeom_p": p_hyper,
    }
    (RES / "cond_CASE.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[case] cond_CASE.json written: {gene}", flush=True)


if __name__ == "__main__":
    main()
