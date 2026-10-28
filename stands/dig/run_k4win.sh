#!/bin/bash
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
L=/home/trim/wblogs/k4sel.log; echo "== window $(date)" >> $L
for x in "0.25 10" "0.25 13" "0.1 10" "0.25 7"; do set -- $x
 r=$(SEL=sens FRAC=$1 D21LMIN=$2 nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen k4sel 2>&1 | grep -E "MEAN mse_final|Error" | tail -1)
 echo "SEL=sens FRAC=$1 LMIN=$2 | $r" >> $L
done
echo DONE_WIN >> $L
