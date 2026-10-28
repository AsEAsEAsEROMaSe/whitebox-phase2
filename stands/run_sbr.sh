#!/bin/bash
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
E=/home/trim/wb2lean/estimator_v30_sbr.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
run() { name=$1; shift; env "$@" /home/trim/fs16/bin/whest run --estimator $E --dataset $DS \
  --split mini --n-mlps 2 --flop-budget 2199023255552 --no-residual-wall-time-limit \
  --wall-time-limit 900 --max-threads 1 --format json \
  > /home/trim/wblogs/sbr_$name.json 2>/dev/null; echo "$name done $(date +%H:%M)" >> /home/trim/wblogs/sbr.log; }
E=/home/trim/whest-p2-cumulant-k3/estimators/estimator_v29.py
run y_base32
run y_min16 V26_STRASSEN_MIN=16
E=/home/trim/wb2lean/estimator_v30_sbr2.py
run y_m16s3 V30_SBR=1 V30_SBR_LEV=3 V26_STRASSEN_MIN=16
run y_m16s4 V30_SBR=1 V30_SBR_LEV=4 V26_STRASSEN_MIN=16
run y_m12s5 V30_SBR=1 V26_STRASSEN_MIN=12
echo DONE_SBR >> /home/trim/wblogs/sbr.log
