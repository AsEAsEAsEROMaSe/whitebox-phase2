#!/bin/bash
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 LEAN_DEV=cuda
cd /home/trim/wb2lean
L=/home/trim/wblogs/lean_d21.log; echo "== start $(date)" >> $L
PY=/home/trim/notqa-venv/bin/python3
run() { tag=$1; md=$2; shift 2; echo "== $tag" >> $L; env "$@" $PY -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen $md 2>&1 | grep -v Warning | grep -E "dump|D21~|MEAN" >> $L; }
run audit rec D21USE=0
run use rec D21USE=1
run use_feed rec D21USE=1 D21FEED=1
run use_feed_k31 k31 D21USE=1 D21FEED=1
run use_feed_r384 rec D21USE=1 D21FEED=1 D21R=384
run use_feed_r128 rec D21USE=1 D21FEED=1 D21R=128
run use_off_only rec D21USE=1 D21FEED=1 D21DIAG=0
run use_feed_w05 rec D21USE=1 D21FEED=1 D21W=0.5
echo DONE_D21 >> $L
