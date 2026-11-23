#!/bin/bash
set -u
cd ~/depplm_diag_20261123
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export DEPPLM_DIAG_LOCK="$PWD/docs/revision_v5/diagnostic_lock_t33.json"
export DEPPLM_DIAG_OUTPUT="$PWD/results/family_inductive_diagnostic_v15/embedding_t33_runs"
mkdir -p "$DEPPLM_DIAG_OUTPUT"
PY=~/miniconda3/envs/cert311/bin/python
ROSTER=61b04033ccc952c073221f57abf73b0817aebb518f001a26d867a242cf110f38
for f in 0 1 2 3 4; do
  rid="diagnostic-t33-of${f}-ss42-ts42"
  if [ -f "$DEPPLM_DIAG_OUTPUT/$rid/completion.json" ]; then echo "SKIP $rid"; continue; fi
  echo "[$(date -u +%FT%TZ)] START $rid"
  /usr/bin/time -f "ELAPSED_SECONDS %e" "$PY" code/run_family_inductive_embedding_probe.py --identity-percent 30 --outer-fold "$f" --split-seed 42 --training-seed 42 --campaign-roster-sha256 "$ROSTER" --device cuda:0 --run-id "$rid" >> t33_probe.log 2>&1
  echo "[$(date -u +%FT%TZ)] DONE $rid rc=$?"
done
echo T33_DONE
