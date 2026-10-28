#!/bin/bash
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/trim/fs16/bin/whest run --estimator /tmp/claude-1000/-mnt-win-d-llm-White-Box/676ecd46-fa7d-4f1c-bdc4-6fa191e9947d/scratchpad/sub31/x/estimator.py --dataset $DS \
  --split mini --n-mlps 2 --flop-budget 2199023255552 --no-residual-wall-time-limit \
  --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/chk31.json 2>/dev/null
echo DONE_CHK31 >> /home/trim/wblogs/chk31.log
