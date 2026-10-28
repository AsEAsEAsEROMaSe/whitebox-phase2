#!/bin/bash
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
L=/home/trim/wblogs/gauge.log; echo "== $(date)" >> $L
for g in "-0.02,999,0" "-0.04,999,0" "0.02,999,0" "0.003,90,0" "0.003,270,0"
do
  r=$(GAUGE=$g DUMPS=0,1,2,3 nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore dig/equalize.py 2>&1 | grep -E "СЕРЕДНЄ|Error")
  echo "r,θ,softmax=$g | $r" >> $L
done
echo DONE >> $L
