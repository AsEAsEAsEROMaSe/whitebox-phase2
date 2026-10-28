#!/bin/bash
# 18.10.2026 E-A: дамп ознак v39 ПО ОДНІЙ мережі (m10..m59 = mini 10..59; s0..s9 = mini 0..9 відкладені).
# Один процес на 60 мереж ріс до 15 ГБ і впав на охороні RAM, тому по одній (~8–9 ГБ).
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
for i in $(seq 10 59); do
  [ -s /home/trim/wblogs/v37_cd_m$i.json ] && continue
  DSX=/home/trim/wb2data/dsub/m$i ./run2.sh /home/trim/wb2lean/v39fija/estimator_v39_dump.py cd_m$i 1 V39_CORRDUMP=/home/trim/wblogs/corrdump
done
echo DUMP3DONE >> /home/trim/wblogs/v37_runs.log
