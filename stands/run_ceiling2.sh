#!/bin/bash
# 13.10.2026: стеля K=3 з істинною діагоналлю κ₄ (A4=1 за аудитом: нахил 1.00–1.19), після run_ceiling.sh
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID LEAN_DEV=cuda OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/trim/wb2lean
PY=/home/trim/notqa-venv/bin/python3
run() { tag=$1; gpu=$2; md=$3; shift 3; L=/home/trim/wblogs/ceil_$tag.log; echo "== $tag $(date)" > $L
  env CUDA_VISIBLE_DEVICES=$gpu "$@" nice -n 10 $PY -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen $md 2>&1 | grep -v Warning | grep -E "dump|MEAN|Traceback|Error" >> $L; echo DONE >> $L; }
G1=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef; G2=GPU-bff69481-0d35-74f0-8871-68eb429cdc34; G3=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26
# (очікування знято 13.10: обгортка Bash-tool тримала в cmdline текст із назвою і сама себе блокувала)
( run d21_k31_k4 $G1 k31k4 D21USE=1; run k4_only $G1 k4 ) &
( run d21_all $G2 all D21USE=1; run d21_k31_k4_w05 $G2 k31k4 D21USE=1 W4=0.5 ) &
( run d21_k4 $G3 k4 D21USE=1 ) &
wait; echo ALLDONE_CEIL2 >> /home/trim/wblogs/ceil_d21_k31_k4.log
