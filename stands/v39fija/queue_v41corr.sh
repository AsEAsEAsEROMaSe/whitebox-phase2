#!/bin/bash
# 18.10.2026 a2: E-A перевірка корекції (v41corr = v39 FIJA + таблиця step_F87, fit87c на mini 10..59).
# Чекає R416DONE (порядок R416 → v41corr → v40, домовлено з a5) і відсутності whest; пломба ECORR_C=0 на s0 (= fija_s0 біт у біт), далі c=1.0 і c=0.75 на s0..s9.
while true; do
  if grep -q '^R416DONE' /home/trim/wblogs/v37_runs.log && ! pgrep -f "bin/whest run" >/dev/null; then break; fi
  sleep 30
done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
E=/home/trim/wb2lean/v39fija/estimator_v41corr.py
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh $E v41off_s0 1 ECORR_C=0
for s in s0 s1 s2 s3 s4 s5 s6 s7 s8 s9; do DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v41c_$s 1; done
for s in s0 s1 s2 s3 s4 s5 s6 s7 s8 s9; do DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v41c75_$s 1 ECORR_C=0.75; done
echo V41CDONE >> /home/trim/wblogs/v37_runs.log
