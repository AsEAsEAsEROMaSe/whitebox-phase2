#!/bin/bash
# 17.10.2026: перевірка на 6 мережах, по одному прогону
cd /home/trim/wb2lean/v37dr
./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py drhub6 6 V37_DEADROW=1 V37_DEADHUB=1
./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py base6 6 V37_DEADROW=0
echo Q2DONE >> /home/trim/wblogs/v37_runs.log
