#!/bin/bash
# 19.10.2026 white-box-21: сталий стан грейдера на мережах 3+ (набір q4 = s0..s3 в одному процесі).
# v46 (сабміт 336991), v47 (арени + ROWALIGN 32), v47 з ROWALIGN 128 (лише арени; MSE = v46), v48 (v47 + поріг живих рядків 2.85 і 3.2, стовпці на 2.5). Без whest, MemAvailable >= 14 ГБ. Кінець Q4DONE.
L=/home/trim/wblogs/v37_runs.log
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 14 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
D=/home/trim/wb2data/dsub/q4; E=/home/trim/wb2lean/v39fija
DSX=$D ./run2.sh $E/estimator_v47.py q4_v47 4
DSX=$D ./run2.sh $E/estimator_v47.py q4_v47r128 4 V37_ROWALIGN=128
DSX=$D ./run2.sh $E/estimator_v48.py q4_v48t285 4 V48_ROW_TH=2.85
DSX=$D ./run2.sh $E/estimator_v48.py q4_v48t32 4 V48_ROW_TH=3.2
DSX=$D ./run2.sh $E/estimator_v46.py q4_v46 4
echo Q4DONE >> $L
