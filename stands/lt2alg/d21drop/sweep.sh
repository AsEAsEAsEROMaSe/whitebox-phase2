#!/bin/bash
# 22.10.2026 b9: D21 drop sweep on mini 0..7 (GPU 26c6ed1b)
cd /home/trim/wb2lean/lt2alg/d21drop
for cfg in "0 rand" "0.25 rand" "0.5 rand" "0.25 det" "0.5 det"; do
  set -- $cfg
  echo "== P=$1 MODE=$2"
  D21_DROP_P=$1 D21_DROP_MODE=$2 CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 OMP_NUM_THREADS=2 \
    /home/trim/fs123/bin/python3.12 run1.py 0 8 2>/dev/null | grep "^mini\|DROPSTAT"
done
echo SWEEPDONE
