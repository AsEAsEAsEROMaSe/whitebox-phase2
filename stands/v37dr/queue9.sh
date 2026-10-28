#!/bin/bash
# 17.10.2026: перевірка v37 (ROWALIGN=128) проти v36 на 9 мережах (s4 уже є), flopscope 0.12.3, по одній мережі за прогін
cd /home/trim/wb2lean/v37dr
export WB=/home/trim/fs123
for s in s0 s1 s2 s3 s5 s6 s7 s8 s9; do
  DSX=/home/trim/wb2data/dsub/$s ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py v9q_$s 1 V37_DEADROW=1 V37_DEADHUB=1
  DSX=/home/trim/wb2data/dsub/$s ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py v9b_$s 1 V37_DEADROW=0
done
echo Q9DONE >> /home/trim/wblogs/v37_runs.log
