#!/bin/bash
# v34 на офіційному mini-100 (істина 1e9): помережна і пошарова MSE для зіставлення з ознаками мереж (feat_mini100.npy)
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/trim/fs16/bin/whest run --estimator /home/trim/wb2lean/estimator_v34_m12.py --dataset $DS --split mini --n-mlps 100 --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/v34_100.json 2>/dev/null
echo DONE_V34_100 >> /home/trim/wblogs/v34_100.log
