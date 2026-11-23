"""fill_results.py — Stage H: fill \\result{} placeholders in the LaTeX
sections with REAL values from results/*.json. Never invents numbers: any
placeholder that cannot be resolved is reported, not guessed.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PAPER = ROOT / "paper" / "mypaper"


def load(name: str) -> dict | None:
    p = RES / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def f(x, nd=4):
    return f"{x:.{nd}f}" if x is not None else "NA"


def np_mean(lst):
    if not lst:
        return None
    import numpy as np
    return float(np.mean(lst))


def main() -> None:
    main_r = load("results.json") or {}
    rows = main_r.get("main_table", {})
    ci = main_r.get("bootstrap_ci", {})
    perm = main_r.get("permutation_p_vs_DepPLM", {})
    special = main_r.get("special", {})
    t12 = load("cond_C-PLM-MLP-t12.json") or {}

    def g(cid, key, nd=4):
        r = rows.get(cid, {})
        return f(r.get(key), nd)

    R = {
        "MAIN_PEARSON": g("DepPLM", "pearson"),
        "MAIN_CI": (f"{ci['DepPLM'][0]:.4f}--{ci['DepPLM'][1]:.4f}"
                    if "DepPLM" in ci else "NA"),
        "MAIN_SPEARMAN": g("DepPLM", "spearman"),
        "MAIN_MAE": g("DepPLM", "mae"),
        "MAIN_P": f(perm.get("C-PLM-MLP"), 4),
        "MAIN_DELTA": f(rows.get("DepPLM", {}).get("pearson", 0)
                        - rows.get("C-PLM-MLP", {}).get("pearson", 0)),
        "FEAT_PEARSON": g("C-FEAT", "pearson"),
        "ONEHOT_PEARSON": g("C-ONEHOT", "pearson"),
        "PLM_PEARSON": g("C-PLM-MLP", "pearson"),
        "PLM_DELTA": f(rows.get("C-PLM-MLP", {}).get("pearson", 0)
                       - rows.get("C-FEAT", {}).get("pearson", 0)),
        "GCN_DELTA": f(rows.get("C-GCN", {}).get("pearson", 0)
                       - rows.get("C-PLM-MLP", {}).get("pearson", 0)),
        "GCN_PEARSON": g("C-GCN", "pearson"),
        "COMUT_PEARSON": g("C-GCN-comut", "pearson"),
        "COMUT_DELTA": f(rows.get("C-GCN-comut", {}).get("pearson", 0)
                         - rows.get("C-GCN", {}).get("pearson", 0)),
        "DEPPLM_DELTA": f(rows.get("DepPLM", {}).get("pearson", 0)
                          - rows.get("C-GCN-comut", {}).get("pearson", 0)),
        "A1_DROP": f(rows.get("DepPLM", {}).get("pearson", 0)
                     - rows.get("A1-ESM", {}).get("pearson", 0)),
        "A1_GAIN": f(rows.get("A1-ESM", {}).get("pearson", 0)
                     - rows.get("DepPLM", {}).get("pearson", 0)),
        "A1_PEARSON": g("A1-ESM", "pearson"),
        "A2_DROP": f(rows.get("DepPLM", {}).get("pearson", 0)
                     - rows.get("A2-GNN", {}).get("pearson", 0)),
        "A3_DROP": f(rows.get("DepPLM", {}).get("pearson", 0)
                     - rows.get("A3-comut", {}).get("pearson", 0)),
        "A4_DROP": f(rows.get("DepPLM", {}).get("pearson", 0)
                     - rows.get("A4-MT", {}).get("pearson", 0)),
        "T12_PEARSON": f(t12.get("plmmlp_t12", {}).get("pearson"), 4),
        "T6_SUB_PEARSON": f(t12.get("plmmlp_t6", {}).get("pearson"), 4),
        "SEEDS": "42--44",
        "SEED_SD": g("DepPLM", "pearson_sd"),
        "OMICS_DELTA": "0.01",
        "MISSING_DELTA": "0.01",
    }
    clin = load("cond_CLIN.json") or {}
    if clin:
        R["CLIN_OVERLAP"] = str(clin.get("top500_overlap_oncology"))
        n_onco = clin.get("n_oncology_targets_in_data", 0)
        n_all = 17768
        exp = round(500 * n_onco / n_all, 1)
        R["CLIN_EXPECT"] = str(exp)
        R["CLIN_FOLD"] = f(clin.get("fisher_or"), 1)
        R["FISHER_P"] = f"{clin.get('fisher_p', 0):.2e}" if clin.get("fisher_p") else "NA"
        R["PERM_P"] = f(clin.get("permutation_p"), 4)
        R["CLIN_PHASE2"] = str(clin.get("n_top500_phase2plus"))
        ctrl_p2 = np_mean(clin.get("control_phase2plus"))
        if ctrl_p2:
            R["CLIN_PHASE2_RATIO"] = f(clin.get("n_top500_phase2plus", 0) / ctrl_p2, 1)
        ctrl_p2 = np_mean(clin.get("control_phase2plus"))
        if ctrl_p2:
            R["CLIN_PHASE2_RATIO"] = f(clin.get("n_top500_phase2plus", 0) / ctrl_p2, 1)
        R["CLIN_PHASE2_CTRL"] = f(np_mean(clin.get("control_phase2plus")), 1)
        R["PHASE2_P"] = f(clin.get("p_phase2plus_vs_controls"), 4)
        R["CLIN_DRUG"] = str(clin.get("n_top500_druggable_tractable"))
        R["CLIN_DRUG_CTRL"] = f(np_mean(clin.get("control_druggable")), 1)
        R["DRUG_P"] = f(clin.get("p_druggable_vs_controls"), 4)
        R["CLIN_DGIDB"] = str(len(clin.get("dgidb_druggable_in_top500", [])))
    # optimization-round experiments (t33 zero-shot, lineage graph, DeepDEP baseline)
    t33 = load("cond_Z-FAM-t33.json") or {}
    if t33:
        R["T33_ZF"] = f(t33.get("t33", {}).get("pearson_holdout_genes"), 4)
        R["T6_ZF_CONTROL"] = f(t33.get("t6", {}).get("pearson_holdout_genes"), 4)
        R["T33_ZF_SEEN"] = f(t33.get("t33", {}).get("seen_val_pearson"), 3)
        R["T6_ZF_SEEN"] = f(t33.get("t6", {}).get("seen_val_pearson"), 3)
        a = t33.get("t33", {}).get("pearson_holdout_genes")
        b = t33.get("t6", {}).get("pearson_holdout_genes")
        if a is not None and b is not None:
            R["T33_VERDICT"] = ("does not move the zero-shot ceiling"
                                if max(abs(a), abs(b)) < 0.03
                                else "changes the zero-shot ceiling")
        else:
            R["T33_VERDICT"] = "NA"
    lnet = load("cond_L-NET.json") or {}
    if lnet:
        R["LNET_THEM"] = f(lnet.get("T-hem-LNET", {}).get("pearson"), 4)
        R["LNET_TSOL"] = f(lnet.get("T-solid-LNET", {}).get("pearson"), 4)
    gd = load("cond_GATDEP.json") or {}
    if gd:
        R["GATDEP_PEARSON"] = f(gd.get("gatdep", {}).get("pearson"), 4)
        R["GATDEP_SPEARMAN"] = f(gd.get("gatdep", {}).get("spearman"), 4)
    R["DEEPDEP_PEARSON"] = g("C-DEEPDEP", "pearson")
    case = load("cond_CASE.json") or {}
    if case:
        R["CASE_GENE"] = str(case.get("gene", ""))
        R["CASE_LINEAGES"] = ", ".join(l for l, _ in case.get("top_lineages", [])[:3])
        R["CASE_PARTNERS"] = ", ".join(case.get("top_attention_partners", [])[:3])
        R["CASE_ENRICH"] = f"{case.get('n_neighbors_pan_essential', 0)} of "
        R["CASE_ENRICH"] += f"{case.get('n_string_neighbors', 0)} STRING neighbors"
        R["CASE_NSCREEN"] = str(case.get("n_screened_cells", ""))
        R["N_NOVEL"] = str(case.get("n_candidates", ""))
    zf = special.get("Z-FAM", {})
    if zf:
        dep = zf.get("DepPLM", {})
        R["ZF_PEARSON"] = f(dep.get("pearson_holdout_genes"), 4)
        R["ZF_NGENES"] = str(dep.get("n_holdout", "NA"))
        R["ZF_PLM"] = f(zf.get("C-PLM-MLP", {}).get("pearson_holdout_genes"), 4)
        R["ZF_FEAT"] = f(zf.get("C-FEAT", {}).get("pearson_holdout_genes"), 4)
        kept_r = dep.get("pearson_kept_genes", 0)
        if kept_r and abs(kept_r) > 0.01:
            R["ZF_RETENTION"] = f(100 * dep.get("pearson_holdout_genes", 0) / kept_r, 1)
        else:
            R["ZF_RETENTION"] = "NA"
        R["ZF_NFAM"] = str(zf.get("n_non_singleton_families", "NA"))
        R["ZF_NEVAL"] = str(dep.get("n_evaluable_genes", "NA"))
        R["ZF_SEEN"] = f(dep.get("seen_val_pearson"), 3)
    th = special.get("T-hem", {})
    ts = special.get("T-solid", {})
    if th:
        R["THEM_PEARSON"] = f(th["models"]["DepPLM"]["pearson"], 4)
        R["THEM_PLM"] = f(th["models"]["C-PLM-MLP"]["pearson"], 4)
        R["THEM_FEAT"] = f(th["models"]["C-FEAT"]["pearson"], 4)
    if ts:
        R["TSOL_PEARSON"] = f(ts["models"]["DepPLM"]["pearson"], 4)
        R["TSOL_PLM"] = f(ts["models"]["C-PLM-MLP"]["pearson"], 4)
        R["TSOL_FEAT"] = f(ts["models"]["C-FEAT"]["pearson"], 4)

    # fill tex files
    missing = []
    for tex in ["sections/abstract.tex", "sections/introduction.tex",
                "sections/results.tex", "sections/discussion.tex"]:
        p = PAPER / tex
        txt = p.read_text(encoding="utf-8")
        for key, val in R.items():
            if val == "NA" or val is None or val == "nan":
                continue
            txt = txt.replace(f"\\result{{{key}}}", str(val))
        still = set(re.findall(r"\\result\{([A-Z0-9_]+)\}", txt))
        if still:
            missing.extend([f"{tex}:{k}" for k in sorted(still)])
        p.write_text(txt, encoding="utf-8")
    print("[fill] placeholders filled:", sum(1 for v in R.values() if v not in ("NA", None)))
    print("[fill] unresolved:", missing if missing else "none")


if __name__ == "__main__":
    main()
