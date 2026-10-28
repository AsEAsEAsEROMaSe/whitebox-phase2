#!/bin/bash
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
L=/home/trim/wblogs/gab.log; echo "== $(date)" >> $L
for g in "-0.04,0" "0,0.0017" "-0.04,0.0017" "-0.04,0.001"
do
  r=$(GAB=$g DUMPS=0,1,2,3,4,5,6,7 nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore dig/equalize.py 2>&1 | grep -E "дамп|СЕРЕДНЄ|Error" | tr '\n' ' ')
  echo "a,b=$g | $r" >> $L
done
echo DONE >> $L
