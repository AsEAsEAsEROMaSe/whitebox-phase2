#!/bin/bash
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
L=/home/trim/wblogs/k4b.log; echo "== ser2 $(date)" >> $L
run(){ tag=$1; m=$2; shift 2; r=$(env "$@" nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen $m 2>&1 | grep -E "mse_final|Error" | tr '\n' ' ' | sed 's/\[regen\/[a-z0-9]*\] //g;s/ t=[0-9.]*s//g'); echo "$tag | $r" >> $L; }
run blur_w025 k4b W4=0.25
run blur_w075 k4b W4=0.75
run k4true_L10 k4 D21LMIN=10
run k4true_L13 k4 D21LMIN=13
run blur_w05_L10 k4b W4=0.5 D21LMIN=10
echo DONE2 >> $L
