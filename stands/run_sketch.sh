#!/bin/bash
# чекаємо кінця абляції, тоді A/B дешевого range finder (пломба: SKETCH=0 має дати базу біт у біт)
while ! grep -q DONE_ABLATE /home/trim/wblogs/ablate.log 2>/dev/null; do sleep 20; done
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
E=/home/trim/wb2lean/estimator_v30_sketch.py
run() { name=$1; shift; env "$@" /home/trim/fs16/bin/whest run --estimator $E --dataset $DS \
  --split mini --n-mlps 8 --flop-budget 2199023255552 --no-residual-wall-time-limit \
  --wall-time-limit 900 --max-threads 1 --format json \
  > /home/trim/wblogs/skt_$name.json 2>/dev/null; echo "$name done" >> /home/trim/wblogs/sketch.log; }
run seal V30_SKETCH=0 & run sketch V30_SKETCH=1 & run qpass2 V21_QPASS=2 & wait
echo DONE_SKETCH >> /home/trim/wblogs/sketch.log
