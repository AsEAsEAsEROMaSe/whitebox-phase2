#!/bin/bash
# 21.10.2026 white-box-b9: mid з init від runs/mid на prepA (952 + 1040 нових), 1080 Ti; стартує після MERGEDONE і звільнення карти.
until grep -q MERGEDONE /home/trim/wblogs/prep_mergeA.log; do sleep 5; done
until [ $(nvidia-smi --id=GPU-ddffe75c-cebe-ec59-141d-3a922803464f --query-gpu=memory.used --format=csv,noheader,nounits) -lt 1500 ]; do sleep 5; done
cd /home/trim/wb2lean/lt2alg
env PYTHONPATH=/home/trim/notqa-venv/lib/python3.12/site-packages CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=GPU-ddffe75c-cebe-ec59-141d-3a922803464f /home/trim/fs123/bin/python3.12 lt2_train_seed.py /mnt/gguf/wb_lt2alg/prepA runs/midA \
  --uin --emb 16 --epochs 0 --batch 4 --seq-epochs 6 --seq-lr 1e-3 --e2e-epochs 0 --ntrain 1992 --init runs/mid/lt2_model.pt
