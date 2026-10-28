#!/bin/bash
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda
cd /home/trim/wb2lean
L=/home/trim/wblogs/basis_test.log; echo "== start $(date)" >> $L
for r in 384 256 128; do for b in his cov mix rand; do [ $r = 1024 ] && [ $b != his ] && continue; /home/trim/notqa-venv/bin/python3 -W ignore basis_test.py 0,1 $b $r 2>&1 | grep -E "MEAN|Error|Traceback" >> $L; done; done
echo DONE_BASIS >> $L
