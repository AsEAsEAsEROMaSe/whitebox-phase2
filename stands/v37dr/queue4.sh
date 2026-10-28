#!/bin/bash
# 17.10.2026: по ОДНІЙ мережі за прогін (RAM юзера), v37 (DEADROW+DEADHUB) проти v36
cd /home/trim/wb2lean/v37dr
for s in s2 s3 s4 s5 s6 s7; do
  DSX=/home/trim/wb2data/dsub/$s ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py dh_$s 1 V37_DEADROW=1 V37_DEADHUB=1
  DSX=/home/trim/wb2data/dsub/$s ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py b_$s 1 V37_DEADROW=0
done
echo Q4DONE >> /home/trim/wblogs/v37_runs.log
