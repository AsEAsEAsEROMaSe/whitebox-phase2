#!/bin/bash
# 20.10.2026 white-box-89: breaks.py на мережах 0..2, після ORACCRITDONE, P104 (не 1080). Кінець BREAKSDONE.
O=/home/trim/wblogs
until grep -qx ORACCRITDONE $O/v37_runs.log; do sleep 20; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2; P=/home/trim/notqa-venv/bin/python; S=/home/trim/wb2lean/lt2re/breaks.py
CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 $P $S 0 $O/breaks_n0.json > $O/breaks_n0.log 2>&1 &
CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 $P $S 1 $O/breaks_n1.json > $O/breaks_n1.log 2>&1 &
CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef $P $S 2 $O/breaks_n2.json > $O/breaks_n2.log 2>&1 &
wait
echo BREAKSDONE >> $O/v37_runs.log
