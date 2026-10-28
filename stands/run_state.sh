#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26} LEAN_DEV=cuda OMP_NUM_THREADS=2
cd /home/trim/wb2lean; python3 dumpstate.py "$@" >> /home/trim/wblogs/state_${TAG:-mini}.log 2>&1
