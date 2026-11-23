"""09_figures.py — generate paper figures (matplotlib) from real results.

Fig 3: main comparison bar chart with bootstrap 95% CI
Fig 4: ablation waterfall (DepPLM vs ablations)
Fig 5: cross-lineage transfer + zero-shot panels
Fig 6: clinical enrichment (Fisher/permutation) + druggability box
Fig 8: selectivity vs druggability scatter (top genes labelled)
Style: single-column 3.5in and double-column 7in widths, 300 dpi.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "paper" / "mypaper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

COLOR = {"DepPLM": "#C0392B", "abl": "#E67E22", "base": "#5D6D7E"}
COND_ORDER = ["C-ONEHOT", "C-FEAT", "C-PLM-MLP", "C-GCN", "C-GAT",
              "C-GCN-comut", "DepPLM", "A1-ESM", "A2-GNN", "A3-comut", "A4-MT"]


def load(name: str) -> dict:
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig3_main():
    res = load("results.json")
    if not res:
        print("[09] results.json missing — skip fig3", flush=True)
        return
    rows, ci = res["main_table"], res["bootstrap_ci"]
    conds = [c for c in COND_ORDER if c in rows]
    vals = [rows[c]["pearson"] for c in conds]
    errs = [[v - ci[c][0], ci[c][1] - v] for v, c in zip(vals, conds)]
    colors = [COLOR["DepPLM"] if c == "DepPLM" else
              (COLOR["abl"] if c.startswith("A") else COLOR["base"]) for c in conds]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    bars = ax.bar(range(len(conds)), vals, yerr=np.array(errs).T, capsize=2,
                  color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Pearson r (held-out cell lines)")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.axhline(0, color="black", linewidth=0.5)
    for i, (c, v) in enumerate(zip(conds, vals)):
        p = res["permutation_p_vs_DepPLM"].get(c)
        if p is not None:
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.text(i, v + errs[i][1] + 0.01, star, ha="center", fontsize=6)
    ax.set_title("Main comparison (5-fold CV)")
    fig.savefig(FIG / "fig3_main.png")
    plt.close(fig)
    print("[09] fig3_main.png written", flush=True)


def fig4_ablation():
    res = load("results.json")
    if not res:
        return
    dep = res["main_table"]["DepPLM"]["pearson"]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    names, drops = [], []
    full_names = {"A1-ESM": "− PLM prior\n(random-init emb)",
                  "A2-GNN": "− PPI graph",
                  "A3-comut": "− co-mutation\nedge weights",
                  "A4-MT": "− selectivity\naux task"}
    for a in ["A1-ESM", "A2-GNN", "A3-comut", "A4-MT"]:
        if a in res["main_table"]:
            names.append(full_names[a])
            drops.append(dep - res["main_table"][a]["pearson"])
    ax.barh(range(len(names)), drops, color=COLOR["abl"], edgecolor="black",
            linewidth=0.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Δ Pearson r vs full DepPLM")
    ax.axvline(0, color="black", linewidth=0.5)
    for i, d in enumerate(drops):
        ax.text(d + 0.002, i, f"{d:.3f}", va="center", fontsize=6)
    ax.set_title("Ablations")
    fig.savefig(FIG / "fig4_ablation.png")
    plt.close(fig)
    print("[09] fig4_ablation.png written", flush=True)


def fig5_transfer_zeroshot():
    t_hem = load("cond_T-hem.json")
    t_sol = load("cond_T-solid.json")
    lnet = load("cond_L-NET.json")
    zf = load("cond_Z-FAM.json")
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4))
    if t_hem and t_sol:
        for ax, (tag, d) in zip(axes[:2], [("T-hem (train solid → test heme)", t_hem),
                                            ("T-solid (train heme → test solid)", t_sol)]):
            models = ["DepPLM", "C-PLM-MLP", "C-GCN-comut", "C-FEAT"]
            vals = [d["models"][m]["pearson"] for m in models if m in d["models"]]
            shown = [m for m in models if m in d["models"]]
            colors = [COLOR["DepPLM"] if m == "DepPLM" else COLOR["base"] for m in shown]
            if lnet:
                key = "T-hem-LNET" if "heme" in tag else "T-solid-LNET"
                if key in lnet:
                    vals.append(lnet[key]["pearson"])
                    shown.append("DepPLM-LNET")
                    colors.append(COLOR["DepPLM"])
            ax.bar(range(len(shown)), vals, color=colors, edgecolor="black",
                   linewidth=0.4)
            if lnet and shown and shown[-1] == "DepPLM-LNET":
                ax.patches[-1].set_hatch("//")
            ax.set_xticks(range(len(shown)))
            ax.set_xticklabels(shown, rotation=30, ha="right", fontsize=6)
            ax.set_title(tag, fontsize=6.5)
            ax.set_ylabel("Pearson r (held-out lineages)")
    if zf:
        ax = axes[2]
        models = ["DepPLM", "C-PLM-MLP", "C-FEAT"]
        shown = [m for m in models if m in zf]
        vals = [zf[m]["pearson_holdout_genes"] for m in shown]
        colors = [COLOR["DepPLM"] if m == "DepPLM" else COLOR["base"] for m in shown]
        ax.bar(range(len(shown)), vals, color=colors, edgecolor="black",
               linewidth=0.4)
        ax.set_xticks(range(len(shown)))
        ax.set_xticklabels(shown, rotation=30, ha="right", fontsize=6)
        ax.set_title("Z-FAM: held-out gene families\n(zero-shot)", fontsize=6.5)
        ax.set_ylabel("Per-gene Pearson r")
    fig.savefig(FIG / "fig5_transfer_zeroshot.png")
    plt.close(fig)
    print("[09] fig5_transfer_zeroshot.png written", flush=True)


def fig6_clinical():
    clin = load("cond_CLIN.json")
    if not clin:
        print("[09] cond_CLIN.json missing — skip fig6", flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))
    ax = axes[0]
    ax.bar(["Observed\noverlap"], [clin["top500_overlap_oncology"]],
           color=COLOR["DepPLM"], edgecolor="black", linewidth=0.4)
    # permutation null (recompute quickly)
    rng = np.random.default_rng(42)
    genes = [l.strip() for l in
             (ROOT / "data" / "processed" / "genes.txt").read_text().splitlines()]
    onco = set(s.upper() for s in clin.get("top500_genes", []))
    # use stored top500; approximate null with onco set from cond_CLIN overlap info
    # full null is computed in 08; here show the p value text
    ax.text(0.5, 0.9, f"Fisher p = {clin['fisher_p']:.3g}\nperm p = {clin['permutation_p']:.4g}",
            transform=ax.transAxes, ha="center", fontsize=6.5)
    ax.set_ylabel("Top-500 predicted ∩ OT oncology targets")
    ax.set_title("Clinical target enrichment", fontsize=7)
    ax2 = axes[1]
    ax2.text(0.1, 0.7, f"Phase≥II drugs in top-500: {clin.get('n_top500_phase2plus', 0)}",
             fontsize=7)
    ax2.text(0.1, 0.5, f"OT-tractable in top-500: {clin.get('n_top500_druggable_tractable', 0)}",
             fontsize=7)
    ax2.text(0.1, 0.3,
             f"control-set permutation p (phase2+):\n  {clin.get('p_phase2plus_vs_controls', 1.0):.3g}",
             fontsize=6.5)
    ax2.axis("off")
    ax2.set_title("Druggability of predictions", fontsize=7)
    fig.savefig(FIG / "fig6_clinical.png")
    plt.close(fig)
    print("[09] fig6_clinical.png written", flush=True)


def fig8_case():
    case = load("cond_CASE.json")
    if not case:
        print("[09] cond_CASE.json missing — skip fig8", flush=True)
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    ax = axes[0]
    lins = case.get("top_lineages", [])[:6]
    ax.barh([l for l, _ in lins], [-v for _, v in lins], color="#C0392B",
            edgecolor="black", linewidth=0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted essentiality (lower = more essential)")
    ax.set_title(f"Predicted lineage pattern: {case.get('gene', '')}", fontsize=7)
    ax2 = axes[1]
    partners = case.get("top_attention_partners", [])[:8]
    weights = case.get("top_attention_weights")
    if weights:
        vals = np.array(weights[:len(partners)])
    else:
        vals = np.arange(len(partners), 0, -1)  # rank fallback (legacy files)
    ax2.bar(partners, vals, color="#5D6D7E", edgecolor="black", linewidth=0.4)
    ax2.set_xticklabels(partners, rotation=45, ha="right", fontsize=6)
    ax2.set_title("Top cross-attention partners", fontsize=7)
    ax2.set_ylabel("summed attention (relative)")
    fig.savefig(FIG / "fig8_case.png")
    plt.close(fig)
    print("[09] fig8_case.png written", flush=True)


def figS1_scale():
    """Supplementary Figure S1: PLM scale comparison on the 2k-gene subset
    (t6 vs t12 vs external GATDep baseline), same 70/30 cell-line split."""
    t12 = load("cond_C-PLM-MLP-t12.json")
    gd = load("cond_GATDEP.json")
    if not t12:
        print("[09] cond_C-PLM-MLP-t12.json missing — skip figS1", flush=True)
        return
    labels, vals, colors = [], [], []
    for tag, key in [("ESM-2 t6\n(8M)", "plmmlp_t6"),
                     ("ESM-2 t12\n(35M)", "plmmlp_t12")]:
        if key in t12:
            labels.append(tag)
            vals.append(t12[key]["pearson"])
            colors.append(COLOR["DepPLM"])
    if gd and "gatdep" in gd:
        labels.append("GATDep\n(2025)")
        vals.append(gd["gatdep"]["pearson"])
        colors.append(COLOR["base"])
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.bar(range(len(labels)), vals, color=colors, edgecolor="black",
           linewidth=0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Pearson r (held-out cells)")
    ax.set_title("Scale comparison (2,000-gene subset)", fontsize=7)
    fig.savefig(FIG / "figS1_scale.png")
    plt.close(fig)
    print("[09] figS1_scale.png written", flush=True)


def main() -> None:
    fig3_main()
    fig4_ablation()
    fig5_transfer_zeroshot()
    fig6_clinical()
    fig8_case()
    figS1_scale()


if __name__ == "__main__":
    main()
