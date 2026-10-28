#!/bin/bash
# $1 = GPU uuid, $2 = dump
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 LEAN_DEV=cuda
cd /home/trim/wb2lean; PY=/home/trim/notqa-venv/bin/python3; L=/home/trim/wblogs/birth_err_d$2.log
declare -A ENVS=( [feed]="D21FEED=1" [feedk4]="D21FEED=1 K4FEED=1 K31FEED=1" [useall]="D21USE=1 D21FEED=1 K4FEED=1 K31FEED=1" )
for tag in feed feedk4 useall; do
  echo "== chain $tag dump $2 $(date +%H:%M)" >> $L
  env ${ENVS[$tag]} TAG=$tag $PY -W ignore birth_err.py $2 2>&1 | grep -vE "Warning|warn" >> $L
done
for tag in base feed feedk4 useall; do
  echo "== onestep $tag dump $2 $(date +%H:%M)" >> $L
  if [ $tag = base ]; then H=/home/trim/wb2lean/his_d21_dump%d.npz; else H=/home/trim/wb2lean/his_${tag}_dump%d.npz; fi
  HIS=$H $PY -W ignore onestep.py $2 2>&1 | grep -vE "Warning|warn" | grep -E "РОЗКЛАД" | sed -E 's/.*(l=[ 0-9]+).*ε ([0-9.]+).*(РОЗКЛАД R[^|]*\|[^|]*).*/\1 ε \2 \3/' >> $L
done
echo DONE_BIRTH_$2 >> $L
