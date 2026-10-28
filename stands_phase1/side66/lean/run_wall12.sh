#!/bin/bash
# ЧИСТИЙ послідовний вимір стіни: по ОДНОМУ прогону за раз (RAM ~6 ГБ, не 25)
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
E=/home/trim/whest-p2-cumulant-k3/estimators/estimator_v29.py
for cfg in base min16 min12; do
  case $cfg in base) EX="V26_WARM=1";; min16) EX="V26_STRASSEN_MIN=16";; min12) EX="V26_STRASSEN_MIN=12";; esac
  env $EX /home/trim/fs16/bin/whest run --estimator $E --dataset $DS --split mini --n-mlps 6 \
    --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 \
    --max-threads 1 --format json > /home/trim/wblogs/w_$cfg.json 2>/dev/null
  echo "$cfg done" >> /home/trim/wblogs/wall12.log
done
echo DONE_WALL12 >> /home/trim/wblogs/wall12.log
