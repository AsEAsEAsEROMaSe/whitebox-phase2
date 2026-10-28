#!/bin/bash
# 16.10.2026: екстракція моди масштабу на dense (несене G), дамп 0; спершу база dense
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
P=/home/trim/notqa-venv/bin/python3
echo "== $(date)"
for cfg_e in "dense _" "dense c34f" "dense c34" "dense 3" "regen c" "regen c4f"; do
  set -- $cfg_e; e=$2; [ "$e" = "_" ] && e=""
  echo "$1 EX=$e | $(EXTRACT=$e nice -n 15 $P -W ignore lean_oracle.py 0 $1 x 2>&1 | grep -E 'MEAN|Error|EPS' | tr '\n' ' ') | $(date +%H:%M)"
done
echo ALLDONE
