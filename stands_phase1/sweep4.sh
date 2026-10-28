#!/bin/bash
# СВІП ГЛИБИНИ на pool3, розкиданий по 4 картах у дві хвилі.
# Довшій програмі — ширша пам'ять: інакше регістри перезаписуються і мертвого
# коду стає БІЛЬШЕ (виміряно: при NL=12/TMP=8 живими були 3 інструкції з 12).
cd /mnt/win_d/llm/White-Box/whest-starterkit/evo/gpu/fx
run () {  # $1=GPU $2=NL $3=TMP
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 FX_POOL=pool3 \
  CE_GPU=$1 CE_POP=4096 CE_GEN=40 CE_LEN=$2 CE_TMP=$3 CE_TAG=d$2 \
  /mnt/win_d/llm/White-Box/torch_venv/bin/python -u cuda_evo.py \
  > /mnt/win_d/llm/White-Box/bank_s53/fx_depth_$2.log 2>&1
}
run 0 4  8 & run 1 8  8 & run 2 12 8 & run 3 20 16 & wait
run 0 32 16 & run 1 48 24 & run 2 64 24 & run 3 96 32 & wait
echo "СВІП ЗАВЕРШЕНО"
