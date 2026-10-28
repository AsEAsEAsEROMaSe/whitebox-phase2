#!/bin/bash
# скан ручок v29 на 8 офіційних mini; бал = mean(mse_i * max(0.1, util_i)); 4 процеси паралельно
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
E=/home/trim/whest-p2-cumulant-k3/estimators/estimator_v29.py
run() { name=$1; shift; env "$@" /home/trim/fs16/bin/whest run --estimator $E --dataset $DS --split mini --n-mlps 8 --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/knob_$name.json 2>/dev/null; echo "$name done" >> /home/trim/wblogs/knobs_v29.log; }
run age3 V21_AGE_OLD=3 & run age5 V21_AGE_OLD=5 & run rold320 V21_R_OLD=320 & run rold2_192 V24_R_OLD2=192 & wait
run age2_6 V24_AGE_OLD2=6 & run lam100 V17_LAM_SCALE=1.0 & run lam090 V17_LAM_SCALE=0.90 & run beta0 V25_BETA=0 & wait
run rold448 V21_R_OLD=448 & run rold2_256 V24_R_OLD2=256 & run age3_rold320 V21_AGE_OLD=3 V21_R_OLD=320 & run age2_8 V24_AGE_OLD2=8 & wait
echo DONE_KNOBS >> /home/trim/wblogs/knobs_v29.log
