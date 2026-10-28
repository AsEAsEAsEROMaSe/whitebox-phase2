#!/bin/bash
# 19.10.2026 white-box-8c: дрібний крок ручок воріт (крок 0.02 дав MSE 1e-3..1e-1, поза квадратичною зоною): GAM ±3e-4 з шару 1. Після V50FDONE. Кінець GAMFDONE.
L=/home/trim/wblogs/v37_runs.log
until grep -q V50FDONE $L; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 10 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
D=/home/trim/wb2data/dsub/q4; E=/home/trim/wb2lean/v39fija
DSX=$D ./run2.sh $E/estimator_v52.py q4_v52gp 4 V52_GAM=1.0003 V52_GLMIN=1
DSX=$D ./run2.sh $E/estimator_v52.py q4_v52gm 4 V52_GAM=0.9997 V52_GLMIN=1
echo GAMFDONE >> $L
