#!/bin/bash
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-GPU-eaa8132f-e46f-2090-d943-0481fbe03aef} OMP_NUM_THREADS=2
cd /home/trim/wb2lean; python3 gen_truth.py ${START:-0} ${END:-1000} ${NS:-4e6} >> /home/trim/wblogs/gen_truth_${TAG:-a}.log 2>&1
