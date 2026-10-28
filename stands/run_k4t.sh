#!/bin/bash
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 LEAN_DEV=cuda OMP_NUM_THREADS=2
cd /home/trim/wb2lean; tag=$2; shift 2
r=$(env "$@" /home/trim/notqa-venv/bin/python3 -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen x 2>&1 | grep -E "MEAN|Error|error" | tail -2); echo "$tag | $r" >> /home/trim/wblogs/k4t.log
