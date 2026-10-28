#!/bin/bash
while ! grep -q DONE_MISS /home/trim/wblogs/miss.log 2>/dev/null; do sleep 20; done
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
E=/home/trim/whest-p2-cumulant-k3/estimators/estimator_v29.py
run() { name=$1; shift; env "$@" /home/trim/fs16/bin/whest run --estimator $E --dataset $DS \
  --split mini --n-mlps 8 --flop-budget 2199023255552 --no-residual-wall-time-limit \
  --wall-time-limit 900 --max-threads 1 --format json \
  > /home/trim/wblogs/mis_$name.json 2>/dev/null; echo "$name done" >> /home/trim/wblogs/miss.log; }
run hub6 V26_STRASSEN_HUB=6 & run sb6 V28_STRASSEN_SB=6 & run fuse99 V28_STRASSEN_FUSE_P=99 & run min16hub V26_STRASSEN_MIN=16 V26_STRASSEN_HUB=6 & wait
echo DONE_MISS3 >> /home/trim/wblogs/miss.log
