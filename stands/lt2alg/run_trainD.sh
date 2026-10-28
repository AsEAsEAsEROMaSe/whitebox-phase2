#!/bin/bash
# 21.10.2026 white-box-b9: prepB = 952 + усі 1800 нових; mid init runs/mid, P104 bff69481.
cd /home/trim && /home/trim/fs123/bin/python3.12 /home/trim/wb2lean/lt2alg/prep_merge.py /home/trim/wb2data/lt2alg_prep /mnt/gguf/wb_lt2alg/dump2 /mnt/gguf/wb_lt2alg/prepB > /home/trim/wblogs/prep_mergeB.log 2>&1
grep -q MERGEDONE /home/trim/wblogs/prep_mergeB.log || exit 1
M=$(python3 -c "import json;print(len(json.load(open('/mnt/gguf/wb_lt2alg/prepB/full_meta.json'))['idx']))")
cd /home/trim/wb2lean/lt2alg
env PYTHONPATH=/home/trim/notqa-venv/lib/python3.12/site-packages CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 /home/trim/fs123/bin/python3.12 lt2_train_seed.py /mnt/gguf/wb_lt2alg/prepB runs/midD \
  --uin --emb 16 --epochs 0 --batch 4 --seq-epochs 5 --seq-lr 1e-3 --e2e-epochs 0 --ntrain $M --init runs/mid/lt2_model.pt
