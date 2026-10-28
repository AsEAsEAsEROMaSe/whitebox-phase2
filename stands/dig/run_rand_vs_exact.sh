#!/bin/bash
# 16.10.2026 ніч: точні парні кумулянти до 4-го порядку (режим all, D21USE=1) проти (А) точних лише на випадковій частці RFRAC
# і (Б) МК-оцінки з MCN семплів. regen, 8 дампів, лише GPU eaa8, один процес.
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
P=/home/trim/notqa-venv/bin/python3; DU=${DUMPS:-0,1,2,3,4,5,6,7}
echo "== $(date) dumps=$DU"
for cfg in ${CFGS:-R:1 R:0.5 R:0.25 R:0.1 M:65536 M:1048576}; do
  IFS=: read k v <<< "$cfg"; set -- $k $v
  if [ "$1" = R ]; then E="RFRAC=$2 MCN=0"; else E="RFRAC=1 MCN=$2"; fi
  echo "$E | $(env $E D21USE=1 nice -n 15 $P -W ignore lean_oracle.py $DU regen all 2>&1 | grep -E 'MEAN mse_final|Error|Trace' | tail -1) | $(date +%H:%M)"
done
echo ALLDONE
