#!/bin/bash
cd /home/trim/wb2lean
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef
LOGN=22 python3 gcore2.py 0 8,15 > /home/trim/wblogs/gcore2_l22.log 2>&1
LOGN=24 python3 gcore2.py 0 15 > /home/trim/wblogs/gcore2_l24.log 2>&1
