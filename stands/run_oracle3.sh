#!/bin/bash
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 LEAN_DEV=cuda
cd /home/trim/wb2lean
L=/home/trim/wblogs/lean_oracle3.log; echo "== start $(date)" >> $L
echo "== regen fit3 (dumps 0,1)" >> $L
/home/trim/notqa-venv/bin/python3 -W ignore lean_oracle.py 0,1 regen fit 2>&1 | grep -v Warning | grep -E "3 моди|MEAN" >> $L
echo "== regen mem (coef з 0,1) на 2..7 і 0,1" >> $L
MEMCOEF=memcoef_0.json,memcoef_1.json /home/trim/notqa-venv/bin/python3 -W ignore lean_oracle.py 2,3,4,5,6,7,0,1 regen mem 2>&1 | grep -v Warning | grep -E "dump|l=15|MEAN" >> $L
echo DONE_ORACLE3 >> $L
