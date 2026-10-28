#!/bin/bash
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1 DUMPS=0,1,2,3,4,5,6,7
nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore dig/layer_corr.py 2>&1 | grep -v "^\[" > /home/trim/wblogs/layer_corr8.log
echo DONE >> /home/trim/wblogs/layer_corr8.log
