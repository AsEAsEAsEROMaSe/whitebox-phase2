#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
cd /home/trim/wb2lean
nice -n 15 /home/trim/notqa-venv/bin/python3 -u -X faulthandler -W ignore seadiag2.py 0 12 > /home/trim/wblogs/seadiag2_0_12.log 2>&1
echo "EXIT $?" >> /home/trim/wblogs/seadiag2_0_12.log
