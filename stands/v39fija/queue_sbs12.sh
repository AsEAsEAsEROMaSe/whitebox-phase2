#!/bin/bash
# 19.10.2026 white-box-16 (колишня 8c): після перезавантаження дораховує SBS_LT 1.2 на sb2 (база q4_sb2, q4_sbs08 є). Кінець SBS12DONE.
L=/home/trim/wblogs/v37_runs.log
while pgrep -f "[b]in/whest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 10 ]; do sleep 30; done
echo "START sbs12 (white-box-16) $(date +%H:%M)" >> $L
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
D=/home/trim/wb2data/dsub/q4; E=/home/trim/wb2lean/v39fija
DSX=$D ./run2.sh $E/estimator_sbs.py q4_sbs12 4 SBS_LT=1.2
echo SBS12DONE >> $L
