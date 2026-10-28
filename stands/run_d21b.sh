#!/bin/bash
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 LEAN_DEV=cuda
cd /home/trim/wb2lean
L=/home/trim/wblogs/lean_d21b.log; echo "== start $(date)" >> $L
PY=/home/trim/notqa-venv/bin/python3
until grep -q DONE_D21C /home/trim/wblogs/lean_d21c.log; do sleep 30; done
run() { tag=$1; md=$2; shift 2; echo "== $tag" >> $L; env "$@" $PY -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen $md 2>&1 | grep -v Warning | grep -E "dump|l= 1 D21|l= 8 D21|l=15 D21|MEAN" | cut -c1-200 >> $L; }
run k4feed rec K4FEED=1
run k4k31feed_k31use k31 K31FEED=1 K4FEED=1
run k31feed_half rec K31FEED=0.5
echo DONE_D21B >> $L
