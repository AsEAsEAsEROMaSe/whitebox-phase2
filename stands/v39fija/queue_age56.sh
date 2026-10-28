#!/bin/bash
# 19.10.2026 white-box-89: AGE_OLD 5 і 6 на sb2 (run_direct, mini 0,1,2; база dir_age4). Кінець AGE56DONE.
# Передбачення: вік 5 raw x0.85..0.95, C/B x1.03..1.06, шанс 35% що бал > x1.02.
L=/home/trim/wblogs/v37_runs.log; E=/home/trim/wb2lean/v39fija; O=/home/trim/wblogs
until [ $(grep -c '^PAUSE' $L) -gt 2 ]; do sleep 10; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 6 ]; do sleep 15; done
echo "START age56 (white-box-89, без whest) $(date +%H:%M)" >> $L
for a in 5 6; do
  V21_AGE_OLD=$a OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/trim/fs123/bin/python3.12 $E/run_direct.py $E/estimator_sb2.py 0,1,2 $O/dir_age$a.json > $O/dir_age$a.log 2>/dev/null
  echo "DONE age56 age$a rc=$? (white-box-89) $(date +%H:%M)" >> $L
done
echo AGE56DONE >> $L
