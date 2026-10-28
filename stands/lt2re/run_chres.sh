#!/bin/bash
# 20.10.2026 white-box-89: chres по одному процесу на мережу, P104 GPU-26c6. Кінець CHRESDONE.
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26
for k in $(seq ${1:-0} ${2:-5}); do /home/trim/fs123/bin/python3.12 /home/trim/wb2lean/lt2re/chres.py $k-$k /home/trim/wblogs/chres_a.json; done
echo CHRESDONE
