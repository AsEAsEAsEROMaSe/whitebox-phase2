#!/bin/bash
# 13.10.2026: стеля K=3 при істинних парних кумулянтах (use-side), 8 офіційних дампів, 3 карти паралельно.
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID LEAN_DEV=cuda OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/trim/wb2lean
PY=/home/trim/notqa-venv/bin/python3
run() { tag=$1; gpu=$2; md=$3; shift 3; L=/home/trim/wblogs/ceil_$tag.log; echo "== $tag $(date)" > $L
  env CUDA_VISIBLE_DEVICES=$gpu "$@" nice -n 10 $PY -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen $md 2>&1 | grep -v Warning | grep -E "dump|MEAN|Traceback|Error" >> $L; echo DONE >> $L; }
G1=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef; G2=GPU-bff69481-0d35-74f0-8871-68eb429cdc34; G3=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26
( run d21_k31 $G1 k31 D21USE=1; run d21_k31_k22 $G1 both D21USE=1 ) &
( run d21_k31_feedall $G2 k31 D21USE=1 D21FEED=1 K31FEED=1 K4FEED=1; run d21_k31_k22_feedall $G2 both D21USE=1 D21FEED=1 K31FEED=1 K4FEED=1 ) &
( run k31_only $G3 k31; run d21_k31_w05 $G3 k31 D21USE=1 W31=0.5 ) &
wait; echo ALLDONE_CEIL >> /home/trim/wblogs/ceil_d21_k31.log
