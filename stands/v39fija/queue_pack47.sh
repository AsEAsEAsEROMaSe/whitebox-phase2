#!/bin/bash
# 18.10.2026 a5: пакет v47 (без сабміту) після STEADYDONE white-box-21, без whest.
L=/home/trim/wblogs/v37_runs.log
until grep -qE '^(STEADYDONE|STOP steady)' $L; do sleep 15; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 10; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 14 ]; do sleep 15; done
/home/trim/wb2lean/v39fija/pack.sh /home/trim/wb2lean/v39fija/estimator_v47.py v47 > /home/trim/wblogs/pack_v47.log 2>&1; echo "PACK v47 rc=$?" >> $L
