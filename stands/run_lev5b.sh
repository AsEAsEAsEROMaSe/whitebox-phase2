#!/bin/bash
# 08.10.2026: ABBA L3/L5/L5/L3, 4 ядра (8-11), nice 10, щоб не забирати весь CPU
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
i=0
for L in 3 5 5 3; do
  i=$((i+1))
  echo "START $i L$L $(date +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> /home/trim/wblogs/lev5b.log
  V30_SBR_LEV=$L nice -n 10 taskset -c 8-11 /home/trim/fs16/bin/whest run --estimator /home/trim/wb2lean/estimator_v34_m12.py --dataset $DS --split mini --n-mlps 2 --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/lev5b_${i}_L$L.json 2>/dev/null
  echo "DONE $i L$L $(date +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> /home/trim/wblogs/lev5b.log
done
echo ALLDONE >> /home/trim/wblogs/lev5b.log
