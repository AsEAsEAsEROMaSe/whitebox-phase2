#!/bin/bash
# 08.10.2026: v34 SBR L3 (база) проти L5, послідовно, ті самі 2 мережі, заради стіни
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for L in 3 5; do
  V30_SBR_LEV=$L /home/trim/fs16/bin/whest run --estimator /home/trim/wb2lean/estimator_v34_m12.py --dataset $DS --split mini --n-mlps 2 --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/lev5_L$L.json 2>/dev/null
  echo "DONE L$L $(date +%T)" >> /home/trim/wblogs/lev5.log
done
