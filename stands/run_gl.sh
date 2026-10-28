#!/bin/bash
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 LEAN_DEV=cuda OMP_NUM_THREADS=2
cd /home/trim/wb2lean; shift; env "$@" /home/trim/notqa-venv/bin/python3 -u -W ignore glow.py 0,1,2,3,4,5,6,7
