#!/bin/bash
# 19.10.2026 white-box-8c: виграш LT2 на sb2 (LT_OFF=1) і сила корекції 0.8/1.2; база q4_sb2, порівняння sb/sboff. Кінець SBSDONE.
L=/home/trim/wblogs/v37_runs.log
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 10 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
D=/home/trim/wb2data/dsub/q4; E=/home/trim/wb2lean/v39fija
DSX=$D ./run2.sh $E/estimator_sb2.py q4_sb2off 4 LT_OFF=1
DSX=$D ./run2.sh $E/estimator_sbs.py q4_sbs08 4 SBS_LT=0.8
DSX=$D ./run2.sh $E/estimator_sbs.py q4_sbs12 4 SBS_LT=1.2
echo SBSDONE >> $L
