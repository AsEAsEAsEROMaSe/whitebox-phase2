#!/bin/bash
# 19.10.2026 white-box-16: чекає маркер $1 у v37_runs.log і відсутність whest, тоді перезапускає 2 робітники дампу.
L=/home/trim/wblogs/v37_runs.log
until grep -qx "$1" $L; do sleep 30; done
while pgrep -f "[b]in/whest run" >/dev/null; do sleep 20; done
echo "START lt2dump resume (white-box-16) $(date +%H:%M)" >> $L
cd /home/trim/wb2lean/lt2re
for w in 0 1; do setsid nohup ./run_dump.sh $w >> /home/trim/wblogs/lt2dump_w$w.log 2>&1 < /dev/null & done
