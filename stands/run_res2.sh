#!/bin/bash
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 LEAN_DEV=cuda
cd /home/trim/wb2lean
L=/home/trim/wblogs/d21_res2.log; PY=/home/trim/notqa-venv/bin/python3
for cfg in "D21LEG=16" "D21LEG=64" "D21LEG=256" "D21LEG=1024" "D21LEG=1024 RRES=1024" "CFG=dense"; do
  echo "== $cfg" >> $L; env $cfg $PY -W ignore d21_res.py 0,3 2>&1 | grep -vE "Warning|warn" | grep -E "res|l= 1 |l= 8 |l=15 " | cut -c1-330 >> $L
done
echo DONE_RES2 >> $L
