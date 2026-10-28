#!/bin/bash
# 19.10.2026 white-box-89: v49b (арени, ROWALIGN 128 = база q4_v47r128) (young-atom cap Susupro) на q4 після Q4DONE. Пломба eps=0 на 1 мережі (= перша мережа q4_v46 біт у біт), далі eps 0.005 і 0.003 на 4 мережах. Кінець V49DONE.
L=/home/trim/wblogs/v37_runs.log
until grep -q "^Q4DONE" $L; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 10 ]; do sleep 30; done
echo "V49 START (white-box-89)" >> $L
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
D=/home/trim/wb2data/dsub/q4; E=/home/trim/wb2lean/v39fija
DSX=$D ./run2.sh $E/estimator_v49b.py q4_v49e0 1 V49_Y_EPS=0 V37_ROWALIGN=128
DSX=$D ./run2.sh $E/estimator_v49b.py q4_v49 4 V37_ROWALIGN=128
DSX=$D ./run2.sh $E/estimator_v49b.py q4_v49e3 4 V49_Y_EPS=0.003 V37_ROWALIGN=128
echo V49DONE >> $L
