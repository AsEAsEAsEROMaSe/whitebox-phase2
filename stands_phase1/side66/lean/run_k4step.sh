#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-GPU-eaa8132f-e46f-2090-d943-0481fbe03aef} LEAN_DEV=cuda OMP_NUM_THREADS=2
cd /home/trim/wb2lean
python3 k4step.py chain ${DUMPS:-0:12} > /home/trim/wblogs/k4step_${TAG:-a}.log 2>&1
echo DONE_K4STEP >> /home/trim/wblogs/k4step_${TAG:-a}.log
