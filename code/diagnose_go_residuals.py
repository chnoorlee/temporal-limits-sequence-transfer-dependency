#!/usr/bin/env python3
"""GO-term enrichment of the largest development residuals.

Genes are ranked by method prediction minus observed pan-context mean. Top and
bottom tails are tested against the background of annotated, measured genes with
one-sided Fisher tests and Benjamini--Hochberg correction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def _bh(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    n = values.size
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty(n)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _load_go(paths: list[Path]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for path in paths:
        frame = pd.read_parquet(path, columns=["approvedSymbol", "go"])
        for symbol, terms in zip(frame["approvedSymbol"], frame["go"]):
            if not isinstance(symbol, str) or not isinstance(terms, (list, tuple, np.ndarray)):
                continue
            selected = {term["id"] for term in terms if isinstance(term, dict) and term.get("aspect") == "P"}
            if selected:
                mapping.setdefault(symbol, set()).update(selected)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-tsv", type=Path, required=True)
    parser.add_argument("--go-parquet", nargs="+", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--tail", type=int, default=500)
    parser.add_argument("--minimum-term-size", type=int, default=10)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.consensus_tsv, sep="\t")
    frame["residual"] = frame["method_prediction"] - frame["observed_gene_mean"]
    frame = frame[np.isfinite(frame["residual"])].copy()
    mapping = _load_go(args.go_parquet)
    annotated = frame["gene"].isin(mapping)
    background_genes = set(frame.loc[annotated, "gene"])
    background = len(background_genes)

    term_to_genes: dict[str, set[str]] = {}
    for gene in background_genes:
        for term in mapping[gene]:
            term_to_genes.setdefault(term, set()).add(gene)

    report: dict[str, object] = {"n_genes": int(len(frame)), "n_annotated": background, "tail": args.tail, "terms": {}}
    for direction, subset in (
        ("overpredicted", frame.nlargest(args.tail, "residual")),
        ("underpredicted", frame.nsmallest(args.tail, "residual")),
    ):
        selected = set(subset["gene"]) & background_genes
        rows = []
        for term, genes in term_to_genes.items():
            if len(genes) < args.minimum_term_size:
                continue
            a = len(selected & genes)
            if a == 0:
                continue
            b = len(selected) - a
            c = len(genes) - a
            d = background - len(selected) - c
            odds, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append({
                "go_id": term, "selected_with_term": a, "selected_total": len(selected),
                "background_with_term": len(genes), "background_total": background,
                "odds_ratio": float(odds) if np.isfinite(odds) else None,
                "p_value": float(p_value),
            })
        if rows:
            p_values = _bh([row["p_value"] for row in rows])
            for row, q_value in zip(rows, p_values):
                row["q_value"] = float(q_value)
            rows.sort(key=lambda row: (row["p_value"], -row["selected_with_term"]))
        report["terms"][direction] = rows[:30]
        pd.DataFrame(rows[:100]).to_csv(args.outdir / f"go_enrichment_{direction}.tsv", sep="\t", index=False)

    (args.outdir / "go_enrichment.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for direction in ("overpredicted", "underpredicted"):
        print(direction)
        for row in report["terms"][direction][:8]:
            print(" ", row["go_id"], row["selected_with_term"], "/", row["selected_total"],
                  "q=", f"{row['q_value']:.2e}", "OR=", row["odds_ratio"])


if __name__ == "__main__":
    main()
