#!/bin/bash
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-GPU-bff69481-0d35-74f0-8871-68eb429cdc34} OMP_NUM_THREADS=2
cd /home/trim/wb2lean; python3 stream_train.py >> /home/trim/wblogs/stream.log 2>&1
