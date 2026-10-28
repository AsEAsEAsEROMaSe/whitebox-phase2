#!/bin/bash
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
P=/home/trim/notqa-venv/bin/python3
for cfg in "lam 0.3" "k4 0.5" "sign 0"; do
  set -- $cfg
  echo "== $1 $2 $(date +%H:%M)"
  nice -n 15 $P -W ignore dig/anti_chain.py 0,1,2,3,4,5,6,7 $1 $2 2>&1 | grep -E "^d[0-9]|^LOO|^MEAN|Error|Traceback"
done
echo ANTIDONE
