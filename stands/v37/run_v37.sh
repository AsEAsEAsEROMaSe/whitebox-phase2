#!/bin/bash
# 16.10.2026: $1 = k1|k0 (закон κ31), $2 тег, $3 n-mlps (1 = logan-fitzgerald); один прогін, CPU, лише з дозволу юзера
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/usr/bin/time -v -o /home/trim/wblogs/v37_$2.time nice -n 10 /home/trim/fs16/bin/whest run --estimator /home/trim/wb2lean/v37/estimator_v37acc_$1.py --dataset $DS --split mini --n-mlps ${3:-1} --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/v37_$2.json 2>/home/trim/wblogs/v37_$2.err
echo "DONE $2 $(date +%T)" >> /home/trim/wblogs/v37.log
