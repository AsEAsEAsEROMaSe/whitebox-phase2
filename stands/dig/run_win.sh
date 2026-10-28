#!/bin/bash
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
L=/home/trim/wblogs/win_oracle.log; echo "== $(date)" >> $L
run(){ tag=$1; shift; r=$(env "$@" nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen all 2>&1 | grep -E "MEAN|Error"); echo "$tag | $r" >> $L; }
run all_full D21USE=1
run last6 D21USE=1 D21LMIN=10
run last4 D21USE=1 D21LMIN=12
run first10 D21USE=1 D21LMAX=9
run last2 D21USE=1 D21LMIN=14
echo DONE >> $L
