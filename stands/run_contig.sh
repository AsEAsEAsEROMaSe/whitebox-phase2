#!/bin/bash
# 08.10.2026: v35 V35_CONTIG=1/0 ABBA, 4 ядра (8-11), nice 10, SBR L3
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
i=0
for C in 1 0 0 1; do
  i=$((i+1))
  echo "START $i C$C $(date +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> /home/trim/wblogs/contig.log
  V35_CONTIG=$C nice -n 10 taskset -c 8-11 /home/trim/fs16/bin/whest run --estimator /home/trim/wb2lean/estimator_v35_contig.py --dataset $DS --split mini --n-mlps 2 --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 --max-threads 1 --format json > /home/trim/wblogs/contig_${i}_C$C.json 2>/home/trim/wblogs/contig_${i}.err
  echo "DONE $i C$C $(date +%T) load $(cut -d' ' -f1 /proc/loadavg) bytes $(stat -c %s /home/trim/wblogs/contig_${i}_C$C.json)" >> /home/trim/wblogs/contig.log
done
echo ALLDONE >> /home/trim/wblogs/contig.log
