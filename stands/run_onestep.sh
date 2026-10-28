#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef
cd /home/trim/wb2lean; L=/home/trim/wblogs/onestep.log
until grep -q "res \] dump3\|dump3 mse" /home/trim/wblogs/d21_save.log; do sleep 15; done; sleep 20
for i in 0 3; do echo "== dump $i $(date)" >> $L; /home/trim/notqa-venv/bin/python3 -W ignore onestep.py $i 2>&1 | grep -v Warning >> $L; done
echo DONE_ONESTEP >> $L
