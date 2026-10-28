#!/bin/bash
# 17.10.2026: ROWALIGN=64 (компроміс між виграшем 32 і стіною 128), sarah, по одному
cd /home/trim/wb2lean/v37dr
DSX=/home/trim/wb2data/dsub/s4 ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py r64_s4 1 V37_DEADROW=1 V37_DEADHUB=1 V37_ROWALIGN=64
DSX=/home/trim/wb2data/dsub/s4 ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py r64b_s4 1 V37_DEADROW=0
echo Q7DONE >> /home/trim/wblogs/v37_runs.log
