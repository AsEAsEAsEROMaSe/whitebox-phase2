#!/bin/bash
# 16.10.2026 вечір: предиктор κ4 поверх ланцюга $CHAIN (reg|dense|g96), 8 дампів: raw → фіт LOO → inject. Лише GPU eaa8, один процес.
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
P=/home/trim/notqa-venv/bin/python3; export CHAIN=${CHAIN:-g96}
echo "== $(date) CHAIN=$CHAIN"
for i in 0 1 2 3 4 5 6 7; do nice -n 15 $P -W ignore dig/stack_pred.py raw $i 2>&1 | grep -E "raw|Error|Trace"; done
nice -n 15 $P -W ignore dig/k4pred_fit.py ${CHAIN}_ 2>&1 | tail -3
for i in 0 1 2 3 4 5 6 7; do nice -n 15 $P -W ignore dig/stack_pred.py inject $i 2>&1 | grep -E "inject|Error|Trace"; done
echo ALLDONE
