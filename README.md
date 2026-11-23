# Reproducibility bundle — temporal limits of sequence transfer

This bundle accompanies the manuscript *Temporal limits of protein-sequence
transfer for cancer dependency prediction under unseen similarity components*
(Briefings in Bioinformatics, revision).

## What is included

- `code/` — analysis, campaign, aggregation and diagnostic scripts.
- `protocols/` — the frozen v1.4 protocol, analysis locks and the diagnostic-scope
  note.
- `results/campaign/` — campaign metrics, consensus predictions and consensus
  evidence arrays for the 24Q4 development campaign.
- `results/diagnostics/` — distance strata, learned coefficients, solver audits,
  temporal diagnostics and the protein-length probe.
- `results/sensitivity/` — 20/40/50% identity sensitivity reports.
- `results/embedding_t33/` — exploratory ESM-2 t33-650M probe report.
- `results/layers/` — ESM-2 t6 hidden-layer (2, 4, 6) probe reports.
- `results/diagnostics/go/` — GO-term enrichment of residual tails.
- `results/diagnostics/lineage/` — lineage-stratified residual probe.
- `results/diagnostics/exploratory/` — wider-grid, James–Stein shrinkage and
  sign-relaxed stack probes.
- `artifacts/per_job_gene_predictions.npz` — per-job outer-test gene-mean
  predictions, component identifiers, distances and learned weights (compact).
- `artifacts/per_gene_consensus.tsv` — consensus predictions and component
  annotations for all 17,768 development genes.
- `manifests/` — input SHA-256 records, fold manifests and component identifiers.
- `SHA256SUMS` — checksums for every file in the bundle.

## What is not included

Raw DepMap CRISPR, expression and mutation matrices, and organoid outcome data,
are not redistributed. They remain subject to the providers' terms. The manifests
record release identifiers and checksums so the inputs can be reconstructed from
the official sources.

## Reproduction outline

1. Obtain DepMap 24Q4 and 26Q3 from the DepMap portal and verify the SHA-256 values
   in `manifests/`.
2. Obtain the UniProt reference proteome and the pinned ESM-2 checkpoint.
3. Build the processed matrices with `code/01_download_data.py` and
   `code/02_preprocess.py`.
4. Recreate the frozen embedding with `code/03_embed_proteins.py`.
5. Rebuild the MMseqs2 component manifests with the scripts referenced in
   `protocols/FAMILY_TRANSFER_PROTOCOL.md`.
6. Run the diagnostic campaign with `code/run_family_inductive_diagnostic.py`
   using the locked grid, then aggregate with
   `code/aggregate_family_inductive_diagnostic.py`. The original 45-job campaign
   and its temporal validator are the primary evidence; the diagnostic re-run is a
   post-review reproduction used for the additional analyses.
7. Run `code/diagnose_family_transfer.py`, `code/diagnose_temporal.py`,
   `code/diagnose_orthogonal_features.py`, `code/diagnose_go_residuals.py`,
   `code/diagnose_lineage_residual.py` and `code/exploratory_shrinkage.py` to
   regenerate the diagnostic tables. Layer embeddings can be regenerated with
   `code/embed_t6_layers.py`; the large layer arrays are not included in this
   bundle.

## Diagnostic provenance

The original campaign ran on the designated host recorded in the v1.4 lock. That
host's per-job archives are no longer reachable. The diagnostic re-run uses the
same inputs, seeds, folds, grids, solver settings and gates on a second Linux GPU
host; the only lock difference is the host identity block. The re-run's agreement
with the locked headline metrics is reported in the manuscript.

## License

Code is released under the MIT License. Derived artifacts are released under
CC BY 4.0 unless a source-data restriction applies. Author approval is required
before publication of the deposit.

## Citation

Repository: https://github.com/chnoorlee/temporal-limits-sequence-transfer-dependency

TODO: author-approved author list and Zenodo archival DOI.
