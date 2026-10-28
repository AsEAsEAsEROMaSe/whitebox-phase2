#!/bin/bash
until grep -q ALLDONE_HUBADJ2 /home/trim/wblogs/hubadj.log; do sleep 20; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
cd /home/trim/wb2lean
for LF in 3 12; do
  echo "START fd2 L$LF $(date +%T)" >> /home/trim/wblogs/hubadj.log
  LF=$LF DUMP=0 nice -n 15 /home/trim/notqa-venv/bin/python3 -u -W ignore hubfd2.py 2>&1 | grep -v Warning >> /home/trim/wblogs/hubadj.log
done
echo ALLDONE_HUBFD2 >> /home/trim/wblogs/hubadj.log
