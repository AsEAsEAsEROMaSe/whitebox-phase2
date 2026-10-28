#!/bin/bash
# 20.10.2026 white-box-89: вужчий базис старих κ3 у sb2 (V21_R_OLD 256/288/320, дефолт 352) — регуляризація як у K=4 на стенді.
# run_direct (без whest, ~2 ГБ), mini 0,1,2; база dir_age4.json (R_OLD 352). Кінець ROLDDONE.
# Передбачення: 320 бал x1.00..1.03; 256 MSE x1.02..1.10, C/B x0.93..0.96; шанс 30% що хоч одне > x1.01.
L=/home/trim/wblogs/v37_runs.log; E=/home/trim/wb2lean/v39fija; O=/home/trim/wblogs
until grep -qx K4ATOMDONE $L; do sleep 15; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 5 ]; do sleep 15; done
echo "START rold (white-box-89, без whest, 1 процес ~2 ГБ) $(date +%H:%M)" >> $L
for r in 320 288 256; do
  V21_R_OLD=$r OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/trim/fs123/bin/python3.12 $E/run_direct.py $E/estimator_sb2.py 0,1,2 $O/dir_rold$r.json > $O/dir_rold$r.log 2>/dev/null
  echo "DONE rold $r rc=$? (white-box-89) $(date +%H:%M)" >> $L
done
echo ROLD3DONE >> $L
