#!/bin/bash
# 21.10.2026 white-box-f9: стенд ν після chprune. Кінець NUDONE.
H=/home/trim/wb2lean/lt2re
until grep -q "CHPRUNEDONE (white-box-f9)" /home/trim/wblogs/v37_runs.log; do sleep 10; done
for k in 1 2 3; do
  CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    timeout 900 /home/trim/fs123/bin/python3.12 $H/nu_feat.py $k /home/trim/wblogs/nu_feat_n$k.json >> /home/trim/wblogs/nu_feat.log 2>&1 || echo "FAIL nu $k" >> /home/trim/wblogs/nu_feat.log
done
echo "NUDONE (white-box-f9) $(date +%H:%M)" >> /home/trim/wblogs/v37_runs.log
