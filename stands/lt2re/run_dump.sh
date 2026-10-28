#!/bin/bash
# 19.10.2026 white-box-16: дамп LT2 на 2 ядрах (дозвіл юзера: «даю добро. ядер 2»). Робітник $1 з 2: mini 0..99, далі корпус 0..4095 (диск: ~3.3 МБ на мережу).
# Шарди корпусу докачуються окремо (fetch). Зупинка: файл /home/trim/wb2data/lt2dump/STOP.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/home/trim/fs123/bin/python3; D=/home/trim/wb2data/lt2dump; S=/home/trim/wb2data/corpus14k
cd /home/trim/wb2lean/lt2re
$P dump_mini.py 0-99 $D/mini --step 2 --off $1
for blk in $(seq 0 64 4095); do
  [ -e $D/STOP ] && exit 0
  until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 6 ]; do sleep 30; done
  $P dump_corpus.py $blk-$((blk+63)) $S $D/corpus --step 2 --off $1
done
