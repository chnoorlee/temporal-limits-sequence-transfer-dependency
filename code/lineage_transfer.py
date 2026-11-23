"""lineage_transfer.py — optimization 4: does a lineage-coexpression network
beat the global STRING graph for cross-lineage transfer?

Loads data/processed/lineage_graph.npz (union of per-lineage co-expression
edges, weight = supporting lineages), builds the normalized adjacency, and
reruns the T-hem/T-solid transfer protocols with DepPLM. Compares against the
STRING-graph DepPLM numbers (results/cond_T-hem.json, cond_T-solid.json).
Writes results/cond_L-NET.json (real numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_experiments import (build_model, build_tokens, pd_factorize,
                             selectivity_labels, train_eval, GRID)
from run_special import load_all

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RES = ROOT / "results"


def main() -> None:
    Y, mut, expr, feat, lin, emb, ei_str, ew_str, adj_str = load_all()
    lg = np.load(DATA / "lineage_graph.npz")
    from models import normalized_adjacency
    ei = torch.from_numpy(lg["edge_index"].astype(np.int64)).t().contiguous()
    ew = torch.from_numpy(lg["edge_weight"].astype(np.float32))
    adj = normalized_adjacency(ei, ew, emb.shape[0])
    print(f"[lnet] lineage graph: {ei.shape[1]} edges, "
          f"support mean {lg['edge_weight'].mean():.2f}", flush=True)

    splits = json.loads((DATA / "splits.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_lin = int(np.max(lin)) + 1
    tokens, token_mask = build_tokens(mut)
    sel_y, _ = selectivity_labels(Y)
    out = {"cond": "L-NET", "n_edges": int(ei.shape[1])}

    # resume support: reload conditions completed in an earlier (interrupted)
    # run from the partial file
    partial_path = RES / "cond_L-NET.partial.json"
    if partial_path.exists():
        prev = json.loads(partial_path.read_text(encoding="utf-8"))
        for k, v in prev.items():
            if k not in ("cond", "n_edges") and isinstance(v, dict):
                out[k] = v
        print(f"[lnet] resumed from partial: {sorted(out.keys())}", flush=True)

    def checkpoint() -> None:
        partial_path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    for cond_id, tr_idx, te_idx in [
        ("T-hem-LNET", np.array(splits["P2_train"], dtype=np.int64),
         np.array(splits["P2_test"], dtype=np.int64)),
        ("T-solid-LNET", np.array(splits["P2_test"], dtype=np.int64),
         np.array(splits["P2_train"], dtype=np.int64))]:
        if cond_id in out:
            print(f"[lnet] {cond_id} already done — skip", flush=True)
            continue
        cfg0 = {"model": "depplm"}
        best = None
        for g in GRID:
            c2 = {**cfg0, **g, "lamb_sel": 0.1}
            va = tr_idx[: int(0.2 * len(tr_idx))]
            m = build_model(c2, emb, adj, ei, ew, mut.shape[1], expr.shape[1],
                            n_lin, Y.shape[0])
            met = train_eval(m, c2, Y, mut, expr, feat, lin, emb, ei, ew, adj,
                             tokens, token_mask, tr_idx, va, sel_y, device,
                             f"{cond_id}-tune", max_epochs=20, patience=5)
            if best is None or met["pearson"] > best[0]:
                best = (met["pearson"], c2)
        m = build_model(best[1], emb, adj, ei, ew, mut.shape[1], expr.shape[1],
                        n_lin, Y.shape[0])
        met = train_eval(m, best[1], Y, mut, expr, feat, lin, emb, ei, ew, adj,
                         tokens, token_mask, tr_idx, te_idx, sel_y, device,
                         cond_id, max_epochs=40, patience=8)
        out[cond_id] = {"pearson": met["pearson"], "spearman": met["spearman"],
                        "mae": met["mae"]}
        checkpoint()
        print(f"[lnet] {cond_id}: pearson {met['pearson']:.4f}", flush=True)

    (RES / "cond_L-NET.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("[lnet] cond_L-NET.json written", flush=True)


if __name__ == "__main__":
    main()
