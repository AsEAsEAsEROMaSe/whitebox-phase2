#!/bin/bash
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
L=/home/trim/wblogs/k31law.log; echo "== $(date)" >> $L
P=/home/trim/notqa-venv/bin/python3
echo "fit | $(nice -n 15 $P -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen k31fit 2>&1 | grep -E 'MEAN|Error' | tail -1)" >> $L
for c in "0,1" "0,2" "0,1,2" "3,1" "3,1,2" "0,1,2,3"; do
 echo "law cols=$c | $(K31COLS=$c nice -n 15 $P -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen k31law 2>&1 | grep -E 'MEAN|Error' | tail -1)" >> $L
done
echo DONE >> $L
