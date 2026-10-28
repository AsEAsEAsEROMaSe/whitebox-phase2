#!/bin/bash
# 18.10.2026 v40 (старий рівень лише в живих нерозпечених рядках): пломба V40_OLDROWS=0 на s0 (= fija_s0 біт у біт),
# далі v40 на s0..s9 і v40 без E-D (V40_OLD_HOT=0) на s0..s9.
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
for s in s0 s1 s2 s3 s4 s5 s6 s7 s8 s9; do
  DSX=/home/trim/wb2data/dsub/$s ./run2.sh /home/trim/wb2lean/v39fija/estimator_v40_oldrows.py v40_$s 1
done
for s in s0 s1 s2 s3 s4 s5 s6 s7 s8 s9; do
  DSX=/home/trim/wb2data/dsub/$s ./run2.sh /home/trim/wb2lean/v39fija/estimator_v40_oldrows.py v40d_$s 1 V40_OLD_HOT=0
done
echo V40DONE >> /home/trim/wblogs/v37_runs.log
