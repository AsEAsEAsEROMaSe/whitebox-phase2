#!/bin/bash
# 18.10.2026 v40 вирок: v40 (E-C+E-D) 0/4 ×0.969. Пломба нового шляху з усіма рядками (поріг 99 = нічого не викидається,
# але ноги формуються через take/scatter) на s0, далі v40d (лише мертві рядки) на s0..s3. Кінець V40DONE.
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh /home/trim/wb2lean/v39fija/estimator_v40_oldrows.py v40all_s0 1 V40_OLD_DEAD=99 V40_OLD_HOT=99
for s in s0 s1 s2 s3; do
  DSX=/home/trim/wb2data/dsub/$s ./run2.sh /home/trim/wb2lean/v39fija/estimator_v40_oldrows.py v40d_$s 1 V40_OLD_HOT=0
done
echo V40DONE >> /home/trim/wblogs/v37_runs.log
