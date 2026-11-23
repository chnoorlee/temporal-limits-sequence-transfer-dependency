"""07_analysis.py — Stage F: statistics over real results.

- loads results/cond_*.json (+ Z-FAM/T-hem/T-solid)
- main comparison table (CSV + MD)
- paired permutation tests (DepPLM vs each baseline, paired by fold)
- bootstrap 95% CI (resampling folds with replacement, 2000 iters)
- outputs results/analysis_report.md + results/results.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

COND_ORDER = ["C-ONEHOT", "C-DEEPDEP", "C-FEAT", "C-PLM-MLP", "C-GCN", "C-GAT",
              "C-GCN-comut", "DepPLM", "A1-ESM", "A2-GNN", "A3-comut", "A4-MT"]


def load_cond(cid: str) -> dict:
    p = RES / f"cond_{cid}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def paired_permutation(a: np.ndarray, b: np.ndarray, n_perm: int = 1000,
                       seed: int = 42) -> float:
    """Two-sided paired permutation test on mean difference (by-fold pairs)."""
    rng = np.random.default_rng(seed)
    d_obs = np.mean(a - b)
    n = len(a)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if abs(np.mean(signs * (a - b))) >= abs(d_obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def bootstrap_ci(vals: np.ndarray, n_iter: int = 2000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(vals), len(vals))
        means.append(np.mean(vals[idx]))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    rows = {}
    for cid in COND_ORDER:
        d = load_cond(cid)
        if d and "folds" in d:
            folds = d["folds"]
            rows[cid] = {
                "pearson": d["mean"]["pearson"],
                "pearson_sd": d["sd"]["pearson"],
                "spearman": d["mean"]["spearman"],
                "mae": d["mean"]["mae"],
                "per_fold_pearson": [f["metrics"]["pearson"] for f in folds],
            }
    dep = rows.get("DepPLM")
    perm_out, ci_out = {}, {}
    if dep is None:
        print("[07] WARN: cond_DepPLM.json missing — permutation tests skipped", flush=True)
    for cid, r in rows.items():
        if cid == "DepPLM" or dep is None:
            continue
        if len(r["per_fold_pearson"]) == len(dep["per_fold_pearson"]):
            p = paired_permutation(np.array(dep["per_fold_pearson"]),
                                   np.array(r["per_fold_pearson"]))
            perm_out[cid] = p
        lo, hi = bootstrap_ci(np.array(r["per_fold_pearson"]))
        ci_out[cid] = [lo, hi]
    if dep:
        lo, hi = bootstrap_ci(np.array(dep["per_fold_pearson"]))
        ci_out["DepPLM"] = [lo, hi]

    # special conditions
    special = {}
    for cid in ["Z-FAM", "T-hem", "T-solid", "C-PLM-MLP-t12", "L-NET",
                "GATDEP"]:
        d = load_cond(cid)
        if d:
            special[cid] = d

    result = {"main_table": rows, "permutation_p_vs_DepPLM": perm_out,
              "bootstrap_ci": ci_out, "special": special}
    (RES / "results.json").write_text(json.dumps(result, indent=1, default=str),
                                      encoding="utf-8")

    lines = ["# analysis_report.md — real results (Stage F)",
             "",
             "## Main comparison (5-fold CV over cell lines, seed 42)",
             "",
             "| condition | pearson | sd | spearman | mae | p vs DepPLM (perm) | 95% CI |",
             "|---|---|---|---|---|---|---|"]
    for cid in COND_ORDER:
        if cid not in rows:
            continue
        r = rows[cid]
        p = perm_out.get(cid)
        ci = ci_out.get(cid)
        lines.append(f"| {cid} | {r['pearson']:.4f} | {r['pearson_sd']:.4f} | "
                     f"{r['spearman']:.4f} | {r['mae']:.4f} | "
                     f"{'—' if p is None else f'{p:.4f}'} | "
                     f"{'—' if ci is None else f'[{ci[0]:.4f}, {ci[1]:.4f}]'} |")
    lines += ["", "## Special conditions", "```json",
              json.dumps(special, indent=1, default=str), "```"]
    (RES / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("[07] analysis_report.md + results.json written", flush=True)


if __name__ == "__main__":
    main()
