#!/bin/bash
# 16.10.2026 вечір: дешеві варіанти предиктора κ4 на ланцюгу $CHAIN: raw2 → stack_var → inject. Лише GPU eaa8, один процес.
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
P=/home/trim/notqa-venv/bin/python3; export CHAIN=${CHAIN:-reg}
export VARS='[["full",1,16,"all",0.25],["L10",10,16,"all",0.25],["L10b3",10,3,"all",0.25],["L10b3y",10,3,"young",0.25],["L10b3s",10,3,"slice",0.25],["L1s",1,16,"slice",0.25],["L10b2s",10,2,"slice",0.25],["L10b3g",10,3,"none",0.25],["L10b3s10",10,3,"slice",0.10]]'
echo "== $(date) CHAIN=$CHAIN"
[ -z "$SKIP_RAW" ] && for i in 0 1 2 3 4 5 6 7; do nice -n 15 $P -W ignore dig/stack_pred2.py raw $i 2>&1 | grep -E "raw2|Error|Trace"; done
nice -n 15 $P -W ignore dig/stack_var.py 2>&1
for v in $(python3 -c "import json,os; print(' '.join('%s:%d:%s' % (a[0],a[1],a[4]) for a in json.loads(os.environ['VARS'])))"); do
  IFS=: read name lmin fr <<< "$v"
  for i in 0 1 2 3 4 5 6 7; do PSUF=$name LMIN=$lmin FRAC=$fr nice -n 15 $P -W ignore dig/stack_pred2.py inject $i 2>&1 | grep -E "LMIN|Error|Trace"; done
done
echo ALLDONE
