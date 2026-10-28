#!/bin/bash
# 19.10.2026 white-box-89: + sb1 (FIJA у ланцюг B: пломба e0 на 1 мережі, 4 мережі). Ланцюг B Susupro (#336537, MIT, sha256 2537…a669, без змін) на q4 (4 мережі): з коректором і LT_OFF=1. Порівняння з q4_v47 / q4_v47r128. Кінець SBDONE.
L=/home/trim/wblogs/v37_runs.log
while pgrep -f "queue_v52.sh|queue_v50f.sh" >/dev/null || pgrep -f "[w]hest run" >/dev/null; do sleep 30; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 10 ]; do sleep 30; done
echo "SB START (white-box-89)" >> $L
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
D=/home/trim/wb2data/dsub/q4; E=/home/trim/wb2lean/v39fija
DSX=$D ./run2.sh $E/estimator_sb.py q4_sb 4
DSX=$D ./run2.sh $E/estimator_sb1.py q4_sb1e0 1 SB1_FIJA=0
DSX=$D ./run2.sh $E/estimator_sb1.py q4_sb1 4
DSX=$D ./run2.sh $E/estimator_sb.py q4_sboff 4 LT_OFF=1
echo SBDONE >> $L
