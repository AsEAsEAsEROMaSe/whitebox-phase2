#!/bin/bash
# 19.10.2026 white-box-89: повтор age3 (перший убив OOMGUARD через RAM юзера) після поточного age2. Кінець AGEDONE.
L=/home/trim/wblogs/v37_runs.log
while pgrep -f "[b]in/whest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 9 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
D=/home/trim/wb2data/dsub/q4; E=/home/trim/wb2lean/v39fija
echo "START age3b (white-box-89) $(date +%H:%M)" >> $L
DSX=$D ./run2.sh $E/estimator_sb2.py q4_age3 4 V21_AGE_OLD=3
echo AGEDONE >> $L
