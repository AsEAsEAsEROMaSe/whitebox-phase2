#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26} OMP_NUM_THREADS=2
cd /home/trim/wb2lean; LOG=/home/trim/wblogs/refsynth_${TAG:-a}.log; N=${N:-256}
for s in ${SEEDS:-1 2}; do for K in ${KS:-3 4}; do F64=0 DUMPF=/home/trim/wb2data/synth/mlp_${N}_%d.npz python3 ref4.py $s $K SIMPLE >> $LOG 2>&1; done; done
echo DONE_REFSYNTH >> $LOG
