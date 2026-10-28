#!/bin/bash
# 17.10.2026: пари мереж по одному прогону, 2 мережі за прогін
cd /home/trim/wb2lean/v37dr
for p in p2 p4 p6; do
  DSX=/home/trim/wb2data/dsub/$p ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py dh_$p 2 V37_DEADROW=1 V37_DEADHUB=1
  DSX=/home/trim/wb2data/dsub/$p ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py b_$p 2 V37_DEADROW=0
done
echo Q3DONE >> /home/trim/wblogs/v37_runs.log
