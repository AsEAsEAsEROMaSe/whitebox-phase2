#!/bin/bash
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
E=/home/trim/whest-p2-cumulant-k3/estimators/estimator_v29.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
run() { name=$1; shift; env V26_STRASSEN_MIN=16 "$@" /home/trim/fs16/bin/whest run --estimator $E --dataset $DS \
  --split mini --n-mlps 2 --flop-budget 2199023255552 --no-residual-wall-time-limit \
  --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/fr_$name.json 2>/dev/null; echo "$name done $(date +%H:%M)" >> /home/trim/wblogs/front.log; }
run a4r256 V21_R_OLD=256 V24_R_OLD2=128 & run a2r256 V21_AGE_OLD=2 V24_AGE_OLD2=5 V21_R_OLD=256 V24_R_OLD2=128 & wait
run a4r192 V21_R_OLD=192 V24_R_OLD2=96 & run a3r256 V21_AGE_OLD=3 V24_AGE_OLD2=6 V21_R_OLD=256 V24_R_OLD2=128 & wait
run a4r128 V21_R_OLD=128 V24_R_OLD2=64 & run a2r192 V21_AGE_OLD=2 V24_AGE_OLD2=5 V21_R_OLD=192 V24_R_OLD2=96 & wait
run a2r384 V21_AGE_OLD=2 V24_AGE_OLD2=5 & run base16 & wait
echo DONE_FRONT >> /home/trim/wblogs/front.log
