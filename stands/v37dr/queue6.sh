#!/bin/bash
# 17.10.2026: стіна v37 (ROWALIGN=128) проти v36 спина до спини, одна мережа (sarah-kelley), по одному прогону
cd /home/trim/wb2lean/v37dr
DSX=/home/trim/wb2data/dsub/s4 ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py w_q_s4 1 V37_DEADROW=1 V37_DEADHUB=1
DSX=/home/trim/wb2data/dsub/s4 ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py w_b_s4 1 V37_DEADROW=0
DSX=/home/trim/wb2data/dsub/s4 ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py w_q2_s4 1 V37_DEADROW=1 V37_DEADHUB=1
echo Q6DONE >> /home/trim/wblogs/v37_runs.log
