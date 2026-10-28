#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-GPU-eaa8132f-e46f-2090-d943-0481fbe03aef} LEAN_DEV=cuda OMP_NUM_THREADS=2
cd /home/trim/wb2lean; LOG=/home/trim/wblogs/bscale_${TAG:-a}.log
for sc in ${SCALES:-1.0 0.9 0.95 1.05 1.1}; do echo "== BIRTH_SCALE $sc LMIN ${LMIN:-0}" >> $LOG; BIRTH_SCALE=$sc BIRTH_SCALE_LMIN=${LMIN:-0} MODES=base python3 injrun.py ${DUMPS:-0,1,2,3,4,5,6,7} 2>&1 | grep "^MEAN" >> $LOG; done
echo DONE_BSCALE >> $LOG
