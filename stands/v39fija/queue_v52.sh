#!/bin/bash
# 19.10.2026 white-box-8c: v52 ручки воріт (GAM, VW) на q4 після FITMLPDONE; база q4_v47 (дефолти = v47 біт у біт). Кінець V52DONE.
L=/home/trim/wblogs/v37_runs.log
until grep -q FITMLPDONE $L; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 10 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
D=/home/trim/wb2data/dsub/q4; E=/home/trim/wb2lean/v39fija
rm -f /home/trim/wblogs/v50diag.jsonl
DSX=$D ./run2.sh $E/estimator_v50d.py q4_v50d 1 V50_DIAG=/home/trim/wblogs/v50diag.jsonl
DSX=$D ./run2.sh $E/estimator_v50.py q4_v50nc 4 ECORR_C=0
DSX=$D ./run2.sh $E/estimator_v52.py q4_v52g102 4 V52_GAM=1.02
DSX=$D ./run2.sh $E/estimator_v52.py q4_v52g098 4 V52_GAM=0.98
DSX=$D ./run2.sh $E/estimator_v52.py q4_v52w099 4 V52_VW=0.99
DSX=$D ./run2.sh $E/estimator_v52.py q4_v52w101 4 V52_VW=1.01
echo V52DONE >> $L
