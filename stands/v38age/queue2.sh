#!/bin/bash
# 18.10.2026: підтвердження l12a6 на s3..s9; бази v37 для s3, s5, s9 перераховуються (раніше вбиті охороною RAM)
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
for s in s4 s6 s7 s8 s3 s5 s9; do
  case $s in s3|s5|s9) DSX=/home/trim/wb2data/dsub/$s ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py v9q_$s 1 V37_DEADROW=1 V37_DEADHUB=1;; esac
  DSX=/home/trim/wb2data/dsub/$s ./run2.sh /home/trim/wb2lean/v38age/estimator_v38_age.py age_l12a6_$s 1 V37_DEADROW=1 V37_DEADHUB=1 V38_L_LATE=12 V38_AGE_LATE=6
done
echo V38DONE2 >> /home/trim/wblogs/v37_runs.log
