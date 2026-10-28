#!/bin/bash
# 19.10.2026 white-box-8c: v50 (v47 + LT2 Susupro) на q4 після V49DONE: з нашим ECORR і без нього; база q4_v47. Далі fitmlp. Кінці V50DONE, FITMLPDONE.
L=/home/trim/wblogs/v37_runs.log
until grep -q V49DONE $L; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 10 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
D=/home/trim/wb2data/dsub/q4; E=/home/trim/wb2lean/v39fija
DSX=$D ./run2.sh $E/estimator_v50.py q4_v50 4
DSX=$D ./run2.sh $E/estimator_v50.py q4_v50nc 4 ECORR_C=0
echo V50DONE >> $L
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
cd $E
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/trim/fs123/bin/python3 -I fitmlp.py > /home/trim/wblogs/fitmlp.log 2>&1
echo "FITMLPDONE rc=$?" >> $L
