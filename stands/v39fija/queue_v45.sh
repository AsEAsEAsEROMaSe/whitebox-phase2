#!/bin/bash
# 18.10.2026 a5 v45 (ранги тонких ніг e30_a): після черги white-box-21 (FIT90DONE або WEXTRACT2 FAILED; прохання white-box-21: fit90 тримає ~1 ГБ ознак)
# і без whest, MemAvailable >= 14 ГБ. Пломба v45off_s0 (16/16 = v43_s0 біт у біт), (4,8) s0..s9, (8,16) і (16,8) s0..s3. Кінець V45DONE.
L=/home/trim/wblogs/v37_runs.log
until grep -qE '^(FIT90DONE|WEXTRACT2 FAILED)' $L; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 14 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
E=/home/trim/wb2lean/v39fija/estimator_v45.py
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh $E v45off_s0 1 V45_R_FB=16 V45_R_RES=16
for s in s0 s1 s2 s3; do DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v45_$s 1; done
for s in s0 s1 s2 s3; do DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v45f8_$s 1 V45_R_FB=8 V45_R_RES=16; done
for s in s0 s1 s2 s3; do DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v45r8_$s 1 V45_R_FB=16 V45_R_RES=8; done
for s in s4 s5 s6 s7 s8 s9; do DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v45_$s 1; done
echo V45DONE >> $L
