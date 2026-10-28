#!/bin/bash
# 17.10.2026: переоблік на flopscope 0.12.3 / whestbench 0.16.2: v36 і v37 (ROWALIGN=128), sarah, по одному
cd /home/trim/wb2lean/v37dr
export WB=/home/trim/fs123
DSX=/home/trim/wb2data/dsub/s4 ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py f123b_s4 1 V37_DEADROW=0
DSX=/home/trim/wb2data/dsub/s4 ./run2.sh /home/trim/wb2lean/v37dr/estimator_v37_dr.py f123q_s4 1 V37_DEADROW=1 V37_DEADHUB=1
echo Q8DONE >> /home/trim/wblogs/v37_runs.log
