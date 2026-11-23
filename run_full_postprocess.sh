#!/bin/bash
set -u
cd ~/depplm_diag_20261123
PY=~/miniconda3/envs/cert311/bin/python
ROOT="$PWD/results/family_inductive_diagnostic_v15"

echo "[$(date -u +%FT%TZ)] waiting for primary campaign"
while ! grep -q "ALL_DONE" campaign_primary_driver.log 2>/dev/null; do sleep 20; done

echo "[$(date -u +%FT%TZ)] primary post-processing"
bash postprocess_primary.sh > postprocess_primary.log 2>&1

echo "[$(date -u +%FT%TZ)] sensitivity campaign"
bash run_sensitivity_campaign.sh > sensitivity_driver.log 2>&1
for thr in 20 40 50; do
  "$PY" code/aggregate_sensitivity_campaign.py \
    --runs-root "$ROOT/sensitivity/identity_${thr}" \
    --outdir "$ROOT/diagnostics/sensitivity_${thr}" \
    --label "identity_${thr}" >> sensitivity_aggregate.log 2>&1
done

echo "[$(date -u +%FT%TZ)] t33 probe"
bash run_t33_probe.sh > t33_driver.log 2>&1
"$PY" code/aggregate_sensitivity_campaign.py \
  --runs-root "$ROOT/embedding_t33_runs" \
  --outdir "$ROOT/diagnostics/embedding_t33" \
  --label "t33" >> t33_aggregate.log 2>&1

echo "[$(date -u +%FT%TZ)] figures"
FD="$ROOT/figure_data"
mkdir -p "$FD"
cp docs/revision_v5/locked_family_transfer_results.json "$FD/family_transfer_results.json"
ln -sf "$ROOT/diagnostics/diagnostics.json" "$FD/diagnostics.json"
ln -sf "$ROOT/diagnostics/weights_per_job.tsv" "$FD/weights_per_job.tsv"
for thr in 20 40 50; do
  mkdir -p "$FD/sensitivity_${thr}"
  cp "$ROOT/diagnostics/sensitivity_${thr}/report.json" "$FD/sensitivity_${thr}/report.json"
done
"$PY" paper_bib_figures.py --data-dir "$FD" --consensus-npz "$ROOT/campaigns/diagnostic-primary-20261123/consensus-evidence.npz" --outdir "$ROOT/figures" >> figures.log 2>&1

echo FULL_POSTPROCESS_DONE
