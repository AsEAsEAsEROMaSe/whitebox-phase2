#!/bin/bash
# 18.10.2026 v39 з R_OLD=416 (як e30_a) на s0..s9: повернути MSE ціною флопів. Старт через 5 хв після DUMP3DONE (пауза для фіту a2).
until grep -q DUMP3DONE /home/trim/wblogs/v37_runs.log; do sleep 30; done; sleep 300
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
for s in s0 s1 s2 s3 s4 s5 s6 s7 s8 s9; do
  DSX=/home/trim/wb2data/dsub/$s ./run2.sh /home/trim/wb2lean/v39fija/estimator_v39_fija.py r416_$s 1 V21_R_OLD=416
done
echo R416DONE >> /home/trim/wblogs/v37_runs.log
