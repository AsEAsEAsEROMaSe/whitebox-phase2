#!/bin/bash
# 21.10.2026 white-box-b9: тест seq-beta (домішка фінальної цілі в локальний лос), mid init від runs/mid, prepA, P104 26c6ed1b після свого працівника дампу (pid 351960).
until grep -q MERGEDONE /home/trim/wblogs/prep_mergeA.log; do sleep 5; done
while ps -p 351960 >/dev/null; do sleep 5; done
cd /home/trim/wb2lean/lt2alg
env PYTHONPATH=/home/trim/notqa-venv/lib/python3.12/site-packages CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 /home/trim/fs123/bin/python3.12 lt2_train_seed.py /mnt/gguf/wb_lt2alg/prepA runs/midB1 \
  --uin --emb 16 --epochs 0 --batch 4 --seq-epochs 5 --seq-lr 1e-3 --seq-beta 1.0 --e2e-epochs 0 --ntrain 1992 --init runs/mid/lt2_model.pt
