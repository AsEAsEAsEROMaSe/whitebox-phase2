#!/bin/bash
# 18.10.2026 a5: юзер дав дозвіл на сабміт v45 («сабміть»). Пакування одразу після RA32DONE, до v45a white-box-21 і v46.
L=/home/trim/wblogs/v37_runs.log
until grep -qE '^(RA32DONE|STOP ra32)' $L; do sleep 15; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 10; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 14 ]; do sleep 15; done
/home/trim/wb2lean/v39fija/pack.sh /home/trim/wb2lean/v39fija/estimator_v45.py v45 > /home/trim/wblogs/pack_v45.log 2>&1; echo "PACK v45 rc=$?" >> $L
