#!/bin/bash
# $1 = тег, $2 = V35_DEAD_TH, $3 = n-mlps
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 V35_DEAD_TH=$2
nice -n 10 /home/trim/fs16/bin/whest run --estimator /home/trim/wb2lean/v35/estimator_v35_dead.py --dataset $DS --split mini --n-mlps ${3:-8} --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/v35_$1.json 2>/home/trim/wblogs/v35_$1.err
echo "DONE $1 $(date +%T)" >> /home/trim/wblogs/v35.log
