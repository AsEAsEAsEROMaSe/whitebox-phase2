#!/bin/bash
# 21.10.2026 white-box-b9: full з init від runs/full на prepA, P104 eaa8132f після кінця свого працівника дампу (pid 351958).
until grep -q MERGEDONE /home/trim/wblogs/prep_mergeA.log; do sleep 5; done
while ps -p 351958 >/dev/null; do sleep 5; done
cd /home/trim/wb2lean/lt2alg
env PYTHONPATH=/home/trim/notqa-venv/lib/python3.12/site-packages CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef /home/trim/fs123/bin/python3.12 lt2_train_seed.py /mnt/gguf/wb_lt2alg/prepA runs/fullA \
  --ext --emb 32 --hidden 128 --uin --epochs 0 --batch 4 --seq-epochs 5 --seq-lr 1e-3 --e2e-epochs 0 --ntrain 1992 --init runs/full/lt2_model.pt
