#!/bin/bash
cd /home/trim/wb2lean
while pgrep -f "dig/run_alpha.sh" >/dev/null; do sleep 15; done
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
L=/home/trim/wblogs/pms.log; echo "== $(date)" >> $L
for a in 0 -0.1 -0.05 0.05 0.1 -0.15; do r=$(ALPHA=$a DUMPS=0,1,2,3 nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore dig/equalize.py 2>&1 | grep -E "СЕРЕДНЄ|Error"); echo "α=$a | $r" >> $L; done
echo DONE >> $L
