#!/bin/bash
# 21.10.2026 b9: optimal gain c on the LT2 correction, held-out mini 48, model $1 on prep $2 ntrain $3
cd /home/trim/wb2lean/lt2alg
CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 OMP_NUM_THREADS=2 \
PYTHONPATH=/home/trim/notqa-venv/lib/python3.12/site-packages /home/trim/fs123/bin/python3.12 lt2_evalc.py $2 runs/evalc_tmp --uin --emb 16 --epochs 0 --seq-epochs 0 --e2e-epochs 0 --ntrain $3 --init $1 $4
