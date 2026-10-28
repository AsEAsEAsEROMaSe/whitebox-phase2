#!/bin/bash
# 18.10.2026: v38 = v37 + довший вік молодих джерел на шарах >= L_LATE; база v9q_s* (v37) уже є; по одному прогону
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
for cfg in "12 6" "12 5" "10 6"; do set -- $cfg
  for s in s0 s1 s2; do
    DSX=/home/trim/wb2data/dsub/$s ./run2.sh /home/trim/wb2lean/v38age/estimator_v38_age.py age_l$1a$2_$s 1 V37_DEADROW=1 V37_DEADHUB=1 V38_L_LATE=$1 V38_AGE_LATE=$2
  done
done
echo V38DONE >> /home/trim/wblogs/v37_runs.log
