#!/bin/bash
# $1 GPU, далі розклади
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 LEAN_DEV=cuda OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /home/trim/wb2lean; shift
for f in "$@"; do r=$(FAC="$f" /home/trim/notqa-venv/bin/python3 -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen x 2>&1 | grep MEAN); echo "$f | $r" >> /home/trim/wblogs/fac.log; done
echo DONE >> /home/trim/wblogs/fac.log
