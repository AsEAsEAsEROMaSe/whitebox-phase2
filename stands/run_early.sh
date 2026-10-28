#!/bin/bash
# 12.10.2026: v36 пошаровий гейт віку (похибка навмисно там, де дешева); послідовно, 1 процес, 4 ядра
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for V in "0 0" "2 8" "2 10" "3 10"; do
  set -- $V; A=$1; LE=$2; tag=a${A}_l${LE}
  echo "START $tag $(date +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> /home/trim/wblogs/early.log
  V36_AGE_EARLY=$A V36_L_EARLY=$LE nice -n 10 taskset -c 8-11 /home/trim/fs16/bin/whest run --estimator /home/trim/wb2lean/estimator_v36_early.py --dataset $DS --split mini --n-mlps 6 --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/early_${tag}.json 2>/home/trim/wblogs/early_${tag}.err
  echo "DONE $tag $(date +%T) bytes $(stat -c %s /home/trim/wblogs/early_${tag}.json)" >> /home/trim/wblogs/early.log
done
echo ALLDONE >> /home/trim/wblogs/early.log
