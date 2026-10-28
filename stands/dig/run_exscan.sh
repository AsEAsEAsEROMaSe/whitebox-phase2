#!/bin/bash
# 16.10.2026: скан сили екстракції c34f на dense, дамп 0 (EXSCALE множить фітовану ε; EXEPS стала ε)
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
P=/home/trim/notqa-venv/bin/python3
echo "== $(date)"
for spec in "c34f S 0.01" "c34f S 0.03" "c34f S -0.01" "c34f S -0.03"; do
  set -- $spec
  if [ "$2" = "S" ]; then EXSCALE=$3; EXEPS=""; else EXSCALE=1; EXEPS=$3; fi
  echo "dense EX=$1 $2=$3 | $(EXTRACT=$1 EXSCALE=$EXSCALE EXEPS=$EXEPS nice -n 15 $P -W ignore lean_oracle.py 0 dense x 2>&1 | grep -E 'MEAN|Error' | tr '\n' ' ')"
done
echo ALLDONE
