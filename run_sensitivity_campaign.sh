#!/bin/bash
set -u
cd ~/depplm_diag_20261123
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/cert311/bin/python
ROSTER=140f51e9ee112215c44b39096bb31b2e6cce4923122774646a893f4c7ab94fb9
for thr in 20 40 50; do
  root="$PWD/results/family_inductive_diagnostic_v15/sensitivity/identity_${thr}"
  mkdir -p "$root"
  export DEPPLM_DIAG_OUTPUT="$root"
  for f in 0 1 2 3 4; do
    rid="diagnostic-sens-i${thr}-of${f}-ss42-ts42"
    if [ -f "$root/$rid/completion.json" ]; then echo "SKIP $rid"; continue; fi
    echo "[$(date -u +%FT%TZ)] START $rid"
    /usr/bin/time -f "ELAPSED_SECONDS %e" "$PY" code/run_family_inductive_diagnostic.py --identity-percent "$thr" --outer-fold "$f" --split-seed 42 --training-seed 42 --campaign-roster-sha256 "$ROSTER" --device cuda:0 --run-id "$rid" >> sensitivity.log 2>&1
    rc=$?
    echo "[$(date -u +%FT%TZ)] DONE $rid rc=$rc"
  done
done
echo SENS_ALL_DONE
