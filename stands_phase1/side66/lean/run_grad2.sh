#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=2
cd /home/trim/wb2lean; for l0 in 13 11; do L0=$l0 python3 gradtest2.py >> /home/trim/wblogs/gradtest2.log 2>&1; done; echo ALLDONE >> /home/trim/wblogs/gradtest2.log
