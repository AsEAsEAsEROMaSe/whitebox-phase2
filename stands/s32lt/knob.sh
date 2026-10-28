#!/bin/bash
# usage: knob.sh <gpu-uuid> <nets> "TAG ENV=V" "TAG ENV=V" ...  (послідовно, одна модель на процес)
G=$1; N=$2; shift 2
for c in "$@"; do set -- $c; T=$1; shift
  env "$@" TAG=$T WHICH=alg CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
   /usr/bin/python3 /home/trim/wb2lean/s32lt/t_knob.py $N
done
