#!/bin/bash
# 21.10.2026 white-box-b9: midE = дотренування midA (held-out 0.94965) на prepB (2752), 3 епохи lr 5e-4, 1080 Ti ddffe75c.
cd /home/trim/wb2lean/lt2alg
env PYTHONPATH=/home/trim/notqa-venv/lib/python3.12/site-packages CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=GPU-ddffe75c-cebe-ec59-141d-3a922803464f /home/trim/fs123/bin/python3.12 lt2_train_seed.py /mnt/gguf/wb_lt2alg/prepB runs/midE1 \
  --uin --emb 16 --epochs 0 --batch 4 --seq-epochs 1 --seq-lr 5e-4 --e2e-epochs 0 --ntrain 2752 --init runs/midA/lt2_model.pt
