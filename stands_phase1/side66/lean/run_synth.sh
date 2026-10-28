#!/bin/bash
# синтетика n=256: істина + референс K=3/K=4 (factor) + ланцюг reg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26} LEAN_DEV=cuda OMP_NUM_THREADS=2
cd /home/trim/wb2lean; LOG=/home/trim/wblogs/synth_${TAG:-a}.log; N=${N:-256}; L=${L:-16}
for s in ${SEEDS:-1 2}; do
  python3 synth256.py $N $L $s ${LOGN:-27} >> $LOG 2>&1
  for K in 3 4; do F64=0 DUMPF=/home/trim/wb2data/synth/mlp_${N}_%d.npz python3 ref4.py $s $K SIMPLE >> $LOG 2>&1; done
  DUMPF=/home/trim/wb2data/synth/mlp_${N}_%d.npz python3 chainrun.py $s >> $LOG 2>&1
done
echo DONE_SYNTH >> $LOG
