#!/bin/bash
# 21.10.2026 white-box-f9: оракул c3/c4 проти прунінгу, мережі 1..3. Кінець CHPRUNEDONE.
H=/home/trim/wb2lean/lt2re; O=/home/trim/wblogs/chprune.jsonl
for k in 1 2 3; do for pr in 1 0; do
  V30_PRUNE=$pr CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    timeout 1800 /home/trim/fs123/bin/python3.12 $H/chprune.py $k $O >> /home/trim/wblogs/chprune.log 2>&1 || echo "FAIL chprune $k $pr" >> /home/trim/wblogs/chprune.log
done; done
echo "CHPRUNEDONE (white-box-f9) $(date +%H:%M)" >> /home/trim/wblogs/v37_runs.log
