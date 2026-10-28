#!/bin/bash
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda
cd /tmp/claude-1000/-mnt-win-d-llm-White-Box/676ecd46-fa7d-4f1c-bdc4-6fa191e9947d/scratchpad/lean
L=/home/trim/wblogs/lean_oracle1.log; echo "== start $(date)" >> $L
for c in "regen rec" "regen k22" "regen k31" "regen both" "dense rec" "dense both"; do
  echo "== $c" >> $L
  /home/trim/notqa-venv/bin/python3 -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 $c 2>&1 | grep -v Warning >> $L
done
echo DONE_ORACLE1 >> $L
