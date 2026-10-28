#!/bin/bash
# 19.10.2026 white-box-89: ранній вхід молодих джерел у базис на sb2: V21_AGE_OLD 3 і 2 (дефолт 4). База q4_sb2. Кінець AGEDONE.
# Передбачення до прогону: AGE 3: флопи x0.88..0.92, MSE x1.00..1.08, бал x0.98..1.08, шанс 35% що > x1.01;
# AGE 2: флопи x0.78..0.85, MSE x1.05..1.25, шанс 20%.
L=/home/trim/wblogs/v37_runs.log
while pgrep -f "[b]in/whest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 8 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
D=/home/trim/wb2data/dsub/q4; E=/home/trim/wb2lean/v39fija
echo "START age3 (white-box-89) $(date +%H:%M)" >> $L
DSX=$D ./run2.sh $E/estimator_sb2.py q4_age3 4 V21_AGE_OLD=3
echo "START age2 (white-box-89) $(date +%H:%M)" >> $L
DSX=$D ./run2.sh $E/estimator_sb2.py q4_age2 4 V21_AGE_OLD=2
echo AGEDONE >> $L
