#!/bin/bash
# скільки наших флопів реально проходить через Штрассена: L0 (щільно) проти L5
while ! grep -q DONE_SKETCH /home/trim/wblogs/sketch.log 2>/dev/null; do sleep 20; done
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
E=/home/trim/whest-p2-cumulant-k3/estimators/estimator_v29.py
run() { name=$1; shift; env "$@" /home/trim/fs16/bin/whest run --estimator $E --dataset $DS \
  --split mini --n-mlps 2 --flop-budget 2199023255552 --no-residual-wall-time-limit \
  --wall-time-limit 900 --max-threads 1 --format json \
  > /home/trim/wblogs/cov_$name.json 2>/dev/null; echo "$name done" >> /home/trim/wblogs/strcov.log; }
run L0 V26_STRASSEN=0 V26_STRASSEN_HUB=0 V28_STRASSEN_SB=0 & run L5 V26_WARM=1 & wait
echo DONE_STRCOV >> /home/trim/wblogs/strcov.log
