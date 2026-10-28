#!/bin/bash
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/trim/fs16/bin/whest run --estimator /home/trim/wb2lean/estimator_v34_m12.py --dataset $DS --split mini --n-mlps 2 --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/v34_a.json 2>/dev/null
echo DONE_V34 >> /home/trim/wblogs/v34.log
