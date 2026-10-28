#!/bin/bash
# 20.10.2026 white-box-89: скан підмножин типів народжень K=4 (AW_SPEC) на мережі 1; які типи можна викинути цілком. Кінець K4TYPESDONE.
# Передбачення: cheap Susupro тримає >= 50% log-виграшу; star і slice4 найважливіші; шанс 40%.
export PYTHONPATH=/home/trim/arcpkg:/home/trim/ext/arc/mlp_cumulant_propagation/src:/home/trim/ext/fc/repo/tools CUDA_DEVICE_ORDER=PCI_BUS_ID AW_DATA=/home/trim/wb2data/k4bed OMP_NUM_THREADS=2 AW_STORE=gpu:4
P=/home/trim/notqa-venv/bin/python; S=/home/trim/wb2lean/lt2re/k4parts.py; O=/home/trim/wblogs/k4types.log
run() { r=$(CUDA_VISIBLE_DEVICES=$1 AW_SPEC="$2" $P $S 256 16 1 1.34e8 2>&1 | grep "k4parts\|Error" | sed 's/.*final //'); echo "$2 $r" >> $O; }
TL=(path star three2); SL=(slice4 slice31 slice22 slice211)
specs=()
for m in $(seq 0 7); do for s in $(seq 0 15); do
  t=""; for i in 0 1 2; do (( m >> i & 1 )) && t="$t+${TL[$i]}"; done
  u=""; for i in 0 1 2 3; do (( s >> i & 1 )) && u="$u+${SL[$i]}"; done
  sp=""; [ -n "$t" ] && sp="${t#+}"; [ -n "$u" ] && sp="${sp:+$sp,}${u#+}"
  [ -n "$sp" ] && specs+=("$sp")
done; done
specs+=("star,slice4+slice31,!slice22+slice211")
a() { for ((k=0;k<${#specs[@]};k+=2)); do run GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 "${specs[$k]}"; done; }
b() { for ((k=1;k<${#specs[@]};k+=2)); do run GPU-bff69481-0d35-74f0-8871-68eb429cdc34 "${specs[$k]}"; done; }
a & b & wait
echo K4TYPESDONE >> /home/trim/wblogs/v37_runs.log
