#!/bin/bash
# 18.10.2026 кандидат v43 (дозвіл юзера «перевір»): пломба v43off_s0 (V40_OLDROWS=0, R_OLD 384 = v41c_s0 біт у біт),
# далі v43 на s0..s9. Кінець V43DONE і CANDDONE (старт дампу white-box-21 m60..m99).
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
E=/home/trim/wb2lean/v39fija/estimator_v43.py
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh $E v43off_s0 1 V40_OLDROWS=0 V21_R_OLD=384
for s in s0 s1 s2 s3 s4 s5 s6 s7 s8 s9; do DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v43_$s 1; done
echo V43DONE >> /home/trim/wblogs/v37_runs.log
echo CANDDONE >> /home/trim/wblogs/v37_runs.log
