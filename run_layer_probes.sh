#!/bin/bash
set -u
cd ~/depplm_diag_20261123
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniconda3/envs/cert311/bin/python
for layer in 2 4 6; do
  export DEPPLM_DIAG_LOCK="$PWD/docs/revision_v5/diagnostic_lock_layer${layer}.json"
  out="$PWD/results/family_inductive_diagnostic_v15/embedding_layer${layer}_runs"
  export DEPPLM_DIAG_OUTPUT="$out"
  mkdir -p "$out"
  roster=$(printf "diagnostic-layer%s-roster-20261123" "$layer" | sha256sum | cut -d' ' -f1)
  for f in 0 1 2 3 4; do
    rid="diagnostic-layer${layer}-of${f}-ss42-ts42"
    if [ -f "$out/$rid/completion.json" ]; then echo "SKIP $rid"; continue; fi
    echo "[$(date -u +%FT%TZ)] START $rid"
    /usr/bin/time -f "ELAPSED_SECONDS %e" "$PY" code/run_family_inductive_diagnostic.py --identity-percent 30 --outer-fold "$f" --split-seed 42 --training-seed 42 --campaign-roster-sha256 "$roster" --device cuda:0 --run-id "$rid" >> layer_probes.log 2>&1
    echo "[$(date -u +%FT%TZ)] DONE $rid rc=$?"
  done
done
echo LAYERS_DONE
