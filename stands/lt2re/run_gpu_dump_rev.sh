#!/bin/bash
# 19.10.2026 white-box-89: третій робітник GPU-дампу на GTX 1080 Ti (юзер: «всі відеокарти можеш юзати»), corpus 4095..0 вниз; кінець рядок DONE gpudump rev.
L=/home/trim/wblogs/v37_runs.log
echo "START gpudump rev (white-box-89, 1080 Ti ddffe75c) $(date +%H:%M)" >> $L
CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-ddffe75c-cebe-ec59-141d-3a922803464f OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/trim/fs123/bin/python3.12 /home/trim/wb2lean/lt2re/gpu_dump.py corpus 0-4095 /home/trim/wb2data/lt2dump/corpus --rev >> /home/trim/wblogs/gpudump_rev.log 2>&1
echo "DONE gpudump rev rc=$? $(date +%H:%M)" >> $L
