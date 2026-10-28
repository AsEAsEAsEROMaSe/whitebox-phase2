#!/bin/bash
# 19.10.2026 white-box-89: AGE_OLD 4/3/2 на sb2 легким прогоном (run_direct.py), mini 0,1,2. Кінець AGEDIRDONE.
L=/home/trim/wblogs/v37_runs.log; E=/home/trim/wb2lean/v39fija; O=/home/trim/wblogs
until [ $(grep -c '^PAUSE' $L) -gt 1 ]; do sleep 10; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 6 ]; do sleep 15; done
echo "START agedir (white-box-89, без whest, 1 процес ~2 ГБ) $(date +%H:%M)" >> $L
for a in 4 3 2; do
  V21_AGE_OLD=$a OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/trim/fs123/bin/python3.12 $E/run_direct.py $E/estimator_sb2.py 0,1,2 $O/dir_age$a.json > $O/dir_age$a.log 2>/dev/null
  echo "DONE agedir age$a rc=$? (white-box-89) $(date +%H:%M)" >> $L
done
echo AGEDIRDONE >> $L
