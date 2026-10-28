#!/bin/bash
# 13.10.2026: κ₄-діагональ: лише масштаб (оракульний нахил по шару) + повний аудит нахилів на 8 дампах
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID LEAN_DEV=cuda OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/trim/wb2lean
PY=/home/trim/notqa-venv/bin/python3
run() { tag=$1; gpu=$2; md=$3; shift 3; L=/home/trim/wblogs/ceil_$tag.log; echo "== $tag $(date)" > $L
  env CUDA_VISIBLE_DEVICES=$gpu "$@" nice -n 10 $PY -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen $md 2>&1 | grep -v Warning | grep -E "dump|MEAN|Traceback|Error" >> $L; echo DONE >> $L; }
G3=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26
run d21_k4s $G3 k4s D21USE=1
run k4s_only $G3 k4s
L=/home/trim/wblogs/ceil_audit_full.log; echo "== audit_full $(date)" > $L
CUDA_VISIBLE_DEVICES=$G3 nice -n 10 $PY -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen rec 2>&1 | grep -v Warning | grep -E "dump|MEAN|wk4_row|Traceback" >> $L; echo DONE >> $L
echo ALLDONE_CEIL3 >> /home/trim/wblogs/ceil_d21_k4s.log
