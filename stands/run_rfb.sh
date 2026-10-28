#!/bin/bash
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
E=/home/trim/wb2lean/estimator_v32_rfb.py
run() { name=$1; shift; env "$@" /home/trim/fs16/bin/whest run --estimator $E --dataset $DS --split mini --n-mlps 2 --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/rfb_$name.json 2>/dev/null; echo "$name done $(date +%H:%M)" >> /home/trim/wblogs/rfb.log; }
run r64 V32_R_FB=64 & run r128 V32_R_FB=128 & wait
echo DONE_RFB >> /home/trim/wblogs/rfb.log
