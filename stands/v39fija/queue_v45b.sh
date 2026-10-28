#!/bin/bash
# 18.10.2026 white-box-21: v45b = v45a з одним видом на хвіст форми (менше тарифікованих reshape). Той самий тест, що дозволив юзер для v45a:
# пломба v45boff_s0 (ROWALIGN 128 = v45_s0 по MSE біт у біт) + v45b s0..s3. Після кінця v45a_s1 (уже йде), без whest, MemAvailable >= 14 ГБ.
# Охорона RAM як у v45a. V45ADONE пишеться безумовно (на нього чекає v46 a5).
L=/home/trim/wblogs/v37_runs.log
until grep -qE '^(DONE|OOMGUARD) v45a_s1' $L; do sleep 15; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 14 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
E=/home/trim/wb2lean/v39fija/estimator_v45b.py
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh $E v45boff_s0 1 V37_ROWALIGN=128
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh $E v45b_s0 1
pk=$(grep '^DONE v45b_s0 ' $L | tail -1 | sed -n 's/.*peakRSS=\([0-9]*\)G.*/\1/p')
if [ -n "$pk" ] && [ "$pk" -lt 13 ]; then
  for s in s1 s2 s3; do DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v45b_$s 1; done
else
  echo "V45B RAM STOP peakRSS=${pk}G" >> $L
fi
echo V45ADONE >> $L
