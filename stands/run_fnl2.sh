#!/bin/bash
# парно, послідовно: v31 (einsum) проти v33 (FASTNL), 2 мережі
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
run() { name=$1; E=$2; shift 2; env "$@" /home/trim/fs16/bin/whest run --estimator $E --dataset $DS \
  --split mini --n-mlps 2 --flop-budget 2199023255552 --no-residual-wall-time-limit \
  --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/fnl2_$name.json 2>/dev/null; echo "$name done $(date +%H:%M)" >> /home/trim/wblogs/fnl2.log; }
run v31a /home/trim/wb2lean/estimator_v31_m16s3.py
run v33a /home/trim/wb2lean/estimator_v33_fastnl.py
run v33b /home/trim/wb2lean/estimator_v33_fastnl.py
run v31b /home/trim/wb2lean/estimator_v31_m16s3.py
echo DONE_FNL2 >> /home/trim/wblogs/fnl2.log
