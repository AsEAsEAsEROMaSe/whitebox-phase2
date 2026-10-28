#!/bin/bash
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 LEAN_DEV=cuda
cd /home/trim/wb2lean
/home/trim/notqa-venv/bin/python3 -W ignore -u threads.py 2>&1 | grep --line-buffered -vE "Warning|warn"
