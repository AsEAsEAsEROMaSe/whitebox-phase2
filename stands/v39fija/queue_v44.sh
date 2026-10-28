#!/bin/bash
# 18.10.2026 v44 (коваріація на живих входах): після PROF43DONE і без whest. Пломба v44off_s0 (V44_COVDEAD=0 = v43_s0 біт у біт),
# далі v44 s0..s9. Кінець V44DONE.
until grep -q "^PROF43DONE" /home/trim/wblogs/v37_runs.log; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
E=/home/trim/wb2lean/v39fija/estimator_v44.py
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh $E v44off_s0 1 V44_COVDEAD=0
for s in s0 s1 s2 s3 s4 s5 s6 s7 s8 s9; do DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v44_$s 1; done
echo V44DONE >> /home/trim/wblogs/v37_runs.log
