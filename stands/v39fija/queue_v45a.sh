#!/bin/bash
# 18.10.2026 white-box-21: v45a (арени пулів Штрассена + ROWALIGN 32) на s0..s3 + пломба v45aoff_s0 (ARENA=1, ROWALIGN=128 = v45_s0 по MSE біт у біт).
# Після RA32DONE (або STOP ra32) і пакування v45 a5 (^PACK v45 rc=), без whest, MemAvailable >= 14 ГБ. V45ADONE пишеться безумовно (на нього чекає v46 a5).
# Охорона RAM: якщо v45a_s0 має peakRSS >= 13G, решту v45a не ганяти (ra32 без арен брав 16G, OOMGUARD).
L=/home/trim/wblogs/v37_runs.log
until grep -qE '^(RA32DONE|STOP ra32)' $L; do sleep 30; done
until grep -qE '^PACK v45 rc=' $L; do sleep 20; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 14 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
E=/home/trim/wb2lean/v39fija/estimator_v45a.py
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh $E v45aoff_s0 1 V37_ROWALIGN=128
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh $E v45a_s0 1
pk=$(grep '^DONE v45a_s0 ' $L | tail -1 | sed -n 's/.*peakRSS=\([0-9]*\)G.*/\1/p')
if [ -n "$pk" ] && [ "$pk" -lt 13 ]; then
  for s in s1 s2 s3; do DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v45a_$s 1; done
else
  echo "V45A RAM STOP peakRSS=${pk}G" >> $L
fi
echo V45ADONE >> $L
