#!/bin/bash
# 18.10.2026 v39 фіксований базис: пломба V39_FIJA=0 на s0 (мусить = age_l12a6_s0 біт у біт), далі v39 на s0..s9
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh /home/trim/wb2lean/v39fija/estimator_v39_fija.py fija0_s0 1 V39_FIJA=0
for s in s0 s1 s2 s3 s4 s5 s6 s7 s8 s9; do
  DSX=/home/trim/wb2data/dsub/$s ./run2.sh /home/trim/wb2lean/v39fija/estimator_v39_fija.py fija_$s 1
done
echo FIJADONE >> /home/trim/wblogs/v37_runs.log
