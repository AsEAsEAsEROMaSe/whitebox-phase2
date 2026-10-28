#!/bin/bash
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda
cd /home/trim/wb2lean
/home/trim/notqa-venv/bin/python3 -W ignore birthdrop.py 0,1,2,3,4,5,6,7 2>&1 | grep -vE "Warning|warn"
