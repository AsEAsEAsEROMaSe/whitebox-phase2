#!/bin/bash
# 17.10.2026: стіна і бал з ROWALIGN=128, спина до спини з базою, по одній мережі
cd /home/trim/wb2lean/v37dr
DSX=/home/trim/wb2data/dsub/s4 ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py q_s4 1 V37_DEADROW=1 V37_DEADHUB=1
DSX=/home/trim/wb2data/dsub/s4 ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py b2_s4 1 V37_DEADROW=0
DSX=/home/trim/wb2data/dsub/s2 ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py q_s2 1 V37_DEADROW=1 V37_DEADHUB=1
echo Q5DONE >> /home/trim/wblogs/v37_runs.log
