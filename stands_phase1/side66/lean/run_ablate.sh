#!/bin/bash
# спадна еластичність: вимикаємо цілі сім'ї членів, міряємо БАЛ (mse*util), не MSE
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
E=/home/trim/whest-p2-cumulant-k3/estimators/estimator_v29.py
run() { name=$1; shift; env "$@" /home/trim/fs16/bin/whest run --estimator $E --dataset $DS \
  --split mini --n-mlps 8 --flop-budget 2199023255552 --no-residual-wall-time-limit \
  --wall-time-limit 900 --max-threads 1 --format json \
  > /home/trim/wblogs/abl_$name.json 2>/dev/null; echo "$name done" >> /home/trim/wblogs/ablate.log; }

run base V26_WARM=1 & run noregen V17_NO_REGEN=1 & run nofeed V17_NO_FEED=1 & run nofb V18_NO_FB=1 & wait
run nowk431 V17_NO_WK431=1 & run nosrclast V19_NO_SRC_LAST=1 & run noconfine V21_NO_CONFINE=1 & run qpass0 V21_QPASS=0 & wait
echo DONE_ABLATE >> /home/trim/wblogs/ablate.log
