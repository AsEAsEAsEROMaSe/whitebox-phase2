#!/bin/bash
# 18.10.2026 a5 v47 = v45b (арени white-box-21) + правки v46: після V46DONE (або STOP v46), без whest, MemAvailable >= 14 ГБ.
# Пломба v47off_s0 (V46_*=0: мусить дорівнювати v45b_s0 біт у біт), далі v47 s0..s9 проти v45. Кінець V47DONE.
L=/home/trim/wblogs/v37_runs.log
until grep -qE '^(V46DONE|STOP v46)' $L; do sleep 30; done
grep -q '^STOP v47' $L && exit 0
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 14 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
E=/home/trim/wb2lean/v39fija/estimator_v47.py
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh $E v47off_s0 1 V46_UEYE=0 V46_TRI=0 V46_NESTG=0 V46_GSUM=0
for s in s0 s1 s2 s3 s4 s5 s6 s7 s8 s9; do grep -q '^STOP v47' $L && break; DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v47_$s 1; done
echo V47DONE >> $L
