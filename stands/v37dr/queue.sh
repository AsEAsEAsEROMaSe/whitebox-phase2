#!/bin/bash
# 17.10.2026: послідовна черга, ОДИН whest run одночасно (RAM юзера). Чекає завершення dr2, далі по черзі.
while ps -p 55093 >/dev/null 2>&1; do sleep 10; done
cd /home/trim/wb2lean/v37dr
./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py base2 2 V37_DEADROW=0
./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py drhub2 2 V37_DEADROW=1 V37_DEADHUB=1
echo QDONE >> /home/trim/wblogs/v37_runs.log
