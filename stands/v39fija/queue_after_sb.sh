#!/bin/bash
# 19.10.2026 white-box-8c: v50f (LT2 з нейтралізованими d21_rowsum/colsum/lam і z у ±4) на q4 після SB2DONE; база q4_v47. Кінець V50FDONE.
L=/home/trim/wblogs/v37_runs.log
until grep -q SB2DONE $L; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 10 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
DSX=/home/trim/wb2data/dsub/q4 ./run2.sh /home/trim/wb2lean/v39fija/estimator_v50f.py q4_v50f 4
echo V50FDONE >> $L
