#!/bin/bash
# синтетика з довільними n, L: істина + K=3 AUGMENT + K=4 AUGMENT + ланцюг
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26} LEAN_DEV=cuda OMP_NUM_THREADS=2
cd /home/trim/wb2lean; LOG=/home/trim/wblogs/synth2_${TAG:-a}.log; N=${N:-128}; L=${L:-8}
for s in ${SEEDS:-1}; do
  python3 synth256.py $N $L $s ${LOGN:-26} 2>&1 | grep -v "Warning\|warn" >> $LOG
  S=$s; [ "$L" != "16" ] && S=$((s+100))
  for K in 3 4; do echo "== n=$N L=$L K=$K AUGMENT" >> $LOG; F64=0 DUMPF=/home/trim/wb2data/synth/mlp_${N}_%d.npz timeout 2400 python3 ref4.py $S $K AUGMENT 2>&1 | grep -v "Warning\|warn" | grep "mse_final\|Error" >> $LOG; done
  DUMPF=/home/trim/wb2data/synth/mlp_${N}_%d.npz python3 chainrun.py $S 2>&1 | grep -v "Warning\|warn" >> $LOG
done
echo DONE_SYNTH2 >> $LOG
