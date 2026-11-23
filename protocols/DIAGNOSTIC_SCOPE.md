# Post-review diagnostic re-run — scope and provenance

Status: diagnostic only; cannot replace the locked v1.4 campaign.

The original 45-job campaign (`family-inductive-v14-formal-r2-20261029`) and its
temporal validator ran on the designated server `192.168.50.7`. That host is not
reachable from the current work environment, and its per-job prediction archives
are not present on the RTX 5090 server.

To answer the major-revision requests for distance-stratified performance,
learned distance-weight profiles, constraint diagnostics and identity-threshold
sensitivity, this directory re-runs the *same* frozen implementation with the
*same* locked inputs on the RTX 5090 server:

- `diagnostic_lock.json` is byte-identical to `docs/revision_v4/family_transfer_lock.json`
  except for `designated_server`, which is rebound to the diagnostic host.
- `code/run_family_inductive_diagnostic.py` is a copy of
  `code/run_family_inductive_experiment.py` with only host-identity, path and
  Windows-path-normalisation changes.
- All input SHA-256 values in the lock are verified unchanged.
- Seeds, grids, cross-fitting, SLSQP settings, endpoints and gates are unchanged.

Outputs are written under `results/family_inductive_diagnostic_v15/` and labelled
`diagnostic-*`. Headline metrics are compared with the locked aggregate; any
discrepancy is reported. The original locked numbers remain the primary evidence
in the manuscript; diagnostic analyses are presented as post hoc and exploratory.

## t33 exploratory probe

The t33-650M embedding artifact predates the residue-chunked generation protocol
and has no recorded pooling manifest. The five-fold probe is therefore labelled a
scale-plus-pooling sensitivity, not a clean model-scale comparison.

## Outcome (2026-11-23)

All 45 primary diagnostic jobs completed and aggregated. The re-run reproduced
the locked development metrics to within 8.7×10⁻¹⁰ (maximum absolute difference)
and the locked 26Q3 cohort and metrics to the same numerical precision. All 6,795
constrained fits were audited with zero failures. Diagnostic outputs are under
`results/family_inductive_diagnostic_v15/` on the diagnostic host and in the
release bundle. They remain post hoc and do not replace the locked campaign.
