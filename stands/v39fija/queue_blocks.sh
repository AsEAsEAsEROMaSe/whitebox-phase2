#!/bin/bash
# 20.10.2026 white-box-89: ціна і цінність кожного блоку sb2 (вимикач проти бази dir_age4.json), бал = raw*max(0.1,C/B), mini 0..2.
# Мета: знайти блоки, що коштують більше, ніж дають у балі. Кінець BLOCKSDONE.
L=/home/trim/wblogs/v37_runs.log; E=/home/trim/wb2lean/v39fija; O=/home/trim/wblogs
echo "START blocks (white-box-89, без whest, 1 процес ~2 ГБ) $(date +%H:%M)" >> $L
for kv in V18_NO_FB=1 V17_NO_REGEN=1 V17_NO_WK431=1 V19_NO_SRC_LAST=1 V21_NO_CONFINE=1 SB1_FIJA=0 AW_R_FB=2 V24_R_OLD2=160 V30_SMIN=16; do
  until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 4 ]; do sleep 15; done
  tag=$(echo $kv | tr '=' '_')
  env $kv OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/trim/fs123/bin/python3.12 $E/run_direct.py $E/estimator_sb2.py 0,1,2 $O/dir_blk_$tag.json > $O/dir_blk_$tag.log 2>/dev/null
  echo "DONE blocks $kv rc=$? (white-box-89) $(date +%H:%M)" >> $L
done
echo BLOCKSDONE >> $L
