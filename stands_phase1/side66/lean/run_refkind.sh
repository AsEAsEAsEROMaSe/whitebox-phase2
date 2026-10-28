#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26} OMP_NUM_THREADS=2
cd /home/trim/wb2lean; LOG=/home/trim/wblogs/refkind_${TAG:-a}.log
for N in ${NS:-128 256 512}; do for s in ${SEEDS:-1}; do for K in ${KS:-3}; do for KIND in ${KINDS:-AUGMENT SIMPLE}; do
  [ -f /home/trim/wb2data/synth/mlp_${N}_${s}.npz ] || continue
  echo "== n=$N seed=$s K=$K $KIND" >> $LOG; F64=0 DUMPF=/home/trim/wb2data/synth/mlp_${N}_%d.npz timeout 1800 python3 ref4.py $s $K $KIND 2>&1 | grep -v "Warning\|warn" | grep "mse_final\|Error" >> $LOG
done; done; done; done
echo DONE_REFKIND >> $LOG
