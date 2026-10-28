#!/bin/bash
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
E=/home/trim/whest-p2-cumulant-k3/estimators/estimator_v29.py
run() { name=$1; shift; env "$@" /home/trim/fs16/bin/whest run --estimator $E --dataset $DS \
  --split mini --n-mlps 8 --flop-budget 2199023255552 --no-residual-wall-time-limit \
  --wall-time-limit 900 --max-threads 1 --format json \
  > /home/trim/wblogs/mis_$name.json 2>/dev/null; echo "$name done" >> /home/trim/wblogs/miss.log; }
run corr0 V17_NO_CORR=0 & run fulllast V19_FULL_LAST=1 & run strnew V26_STRASSEN_NEW=1 & run min16 V26_STRASSEN_MIN=16 & wait
run qp2_1 V24_QPASS2=1 & run qp2_3 V24_QPASS2=3 & run warm0 V26_WARM=0 & run first3 V28_STRASSEN_FIRST=3 & wait
echo DONE_MISS >> /home/trim/wblogs/miss.log
