#!/bin/bash
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
D=${DUMPS:-0,1,2,3,4,5,6,7}
run(){ tag=$1; shift; r=$(env "$@" nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore lean_oracle.py $D regen x 2>&1 | grep -E "mse_final|Error|error" | sed 's/\[regen\/x\] //g;s/ t=[0-9.]*s//g' | tr '\n' ' '); echo "$tag | $r" >> /home/trim/wblogs/dead8.log; }
run hub_th3 HUBDEAD_TH=3
run hub_th2.5 HUBDEAD_TH=2.5
run hub_th2 HUBDEAD_TH=2
run both_th2.5 HUBDEAD_TH=2.5 DEAD_TH=2.5 DEAD_PARTS=k3
echo DONE_HUB8 >> /home/trim/wblogs/dead8.log
