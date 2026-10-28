#!/bin/bash
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
L=/home/trim/wblogs/lin8.log; echo "== $(date)" >> $L
run(){ tag=$1; shift; r=$(env "$@" nice -n 15 /home/trim/notqa-venv/bin/python3 -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen x 2>&1 | grep -E "MEAN|Error"); echo "$tag | $r" >> $L; }
run seal_d2.5 HUBDEAD_TH=2.5
run lin3_d2.5 HUBDEAD_TH=2.5 HUBLIN_TH=3
run lin2.5_d2.5 HUBDEAD_TH=2.5 HUBLIN_TH=2.5
run lin2_d2.5 HUBDEAD_TH=2.5 HUBLIN_TH=2
run lin3.5_d2.5 HUBDEAD_TH=2.5 HUBLIN_TH=3.5
echo DONE >> $L
