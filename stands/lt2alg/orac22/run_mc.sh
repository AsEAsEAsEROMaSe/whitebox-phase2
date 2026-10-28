#!/bin/bash
# 22.10.2026 b9: МК справжніх зрізів (2,2) і (3,1); usage: run_mc.sh <gpu-uuid> <nets...> (дозвіл юзера через c2: «хай ганяють»)
cd /home/trim/wb2lean/lt2alg/orac22
G=$1; shift
for k in "$@"; do
  env PYTHONPATH=/home/trim/notqa-venv/lib/python3.12/site-packages CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2 \
    CUDA_VISIBLE_DEVICES=$G /home/trim/fs123/bin/python3.12 mc22.py $k /mnt/gguf/wb_lt2alg/orac22/mc_n$k.npz 22
done
