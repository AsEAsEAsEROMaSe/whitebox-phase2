#!/bin/bash
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
L=/home/trim/wblogs/k4b.log; echo "== $(date)" >> $L
run(){ tag=$1; m=$2; shift 2; r=$(env "$@" nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen $m 2>&1 | grep -E "MEAN|Error|error" | tail -2); echo "$tag | $r" >> $L; }
run base x
run k4true k4
run k4blur k4b
run k4level k4lev
run k4blur_w05 k4b W4=0.5
echo DONE >> $L
