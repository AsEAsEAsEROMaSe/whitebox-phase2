#!/bin/bash
# 18.10.2026 white-box-21: v45 + V37_ROWALIGN=32 (живі рядки молодого блоку вирівнюються до 32 замість 128) на s0..s3, з виміром стіни.
# Дозвіл юзера 18.10.2026 («даю добро»). Після черги a5 (V45DONE або STOP v45), без whest, MemAvailable >= 14 ГБ. Кінець RA32DONE.
L=/home/trim/wblogs/v37_runs.log
until grep -qE '^(V45DONE|STOP v45)' $L; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 14 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
E=/home/trim/wb2lean/v39fija/estimator_v45.py
for s in s0 s1 s2 s3; do DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v45ra_$s 1 V37_ROWALIGN=32; done
echo RA32DONE >> $L
