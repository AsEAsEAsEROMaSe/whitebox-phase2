#!/bin/bash
# 19.10.2026 white-box-89: ручки на базі sb2 (q4, 4 мережі): R_OLD 416, межа уніонів 12 і 10. Після SB2DONE. Кінець SB3DONE.
L=/home/trim/wblogs/v37_runs.log
until grep -q "^SB2DONE" $L; do sleep 20; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 10 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
D=/home/trim/wb2data/dsub/q4; E=/home/trim/wb2lean/v39fija
DSX=$D ./run2.sh $E/estimator_sb3.py q4_sb3r416 4 V21_R_OLD=416
DSX=$D ./run2.sh $E/estimator_sb3.py q4_sb3c12 4 SB3_LCUT=12
DSX=$D ./run2.sh $E/estimator_sb3.py q4_sb3c10 4 SB3_LCUT=10
echo SB3DONE >> $L
