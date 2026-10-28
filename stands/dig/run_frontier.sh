#!/bin/bash
# 16.10.2026: межа точності всіх методів, 8 дампів, GPU eaa8, один процес
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
echo "== $(date)"
for i in ${1:-0 1 2 3 4 5 6 7}; do nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore dig/frontier.py $i 2>&1 | grep -v "^\s*$"; done
echo ALLDONE
