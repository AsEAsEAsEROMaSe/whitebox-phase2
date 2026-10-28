#!/bin/bash
# 3 дампи паралельно на 3 картах
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID LEAN_DEV=cuda OMP_NUM_THREADS=1
cd /home/trim/wb2lean
i=0
for G in GPU-eaa8132f-e46f-2090-d943-0481fbe03aef GPU-bff69481-0d35-74f0-8871-68eb429cdc34 GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26; do
  CUDA_VISIBLE_DEVICES=$G DUMP=$i LAYERS=3,6,12 NCOL=64 nice -n 15 /home/trim/notqa-venv/bin/python3 -u -W ignore hubfd5.py 2>&1 | grep -v Warning > /home/trim/wblogs/hubfd5_$i.log &
  i=$((i+1))
done
wait
echo ALLDONE_HUBFD5 >> /home/trim/wblogs/hubfd5_0.log
