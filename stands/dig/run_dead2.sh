#!/bin/bash
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
D=${DUMPS:-0,1}
run(){ tag=$1; shift; r=$(env "$@" nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore lean_oracle.py $D regen x 2>&1 | grep -E "MEAN|Error|error" | tr '\n' ' '); echo "$tag | $r" >> /home/trim/wblogs/dead.log; }
run k3_th3 DEAD_TH=3 DEAD_PARTS=k3
run k3_th2 DEAD_TH=2 DEAD_PARTS=k3
run k3_th1 DEAD_TH=1 DEAD_PARTS=k3
run g_th3 DEAD_TH=3 DEAD_PARTS=g
run d4_th3 DEAD_TH=3 DEAD_PARTS=d4
run k3lin_th3 DEAD_TH=3 DEAD_LIN=3 DEAD_PARTS=k3
echo DONE_DEAD2 >> /home/trim/wblogs/dead.log
