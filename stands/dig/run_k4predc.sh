#!/bin/bash
cd /home/trim/wb2lean
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef OMP_NUM_THREADS=1 CUDA_PATH=/usr
for i in 0 1 2 3 4 5 6 7; do nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore dig/k4pred_chain.py $i 20; done
echo DONE
