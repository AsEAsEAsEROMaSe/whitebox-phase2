#!/bin/bash
# стенд 14.10: виключення мертвих нейронів з дорогих частин ланцюга. Лог /home/trim/wblogs/dead.log
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
D=${DUMPS:-0,1}
run(){ tag=$1; shift; r=$(env "$@" nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore lean_oracle.py $D regen x 2>&1 | grep -E "mse_final|Error|error" | tr '\n' ' '); echo "$tag | $r" >> /home/trim/wblogs/dead.log; }
run base X=1
run th99 DEAD_TH=99
run th4 DEAD_TH=4
run th3 DEAD_TH=3
run th3C DEAD_TH=3 DEAD_C=1
run th2.5 DEAD_TH=2.5
run th2 DEAD_TH=2
run ctl_lin3 DEAD_TH=3 DEAD_LIN=3
echo DONE_DEAD >> /home/trim/wblogs/dead.log
