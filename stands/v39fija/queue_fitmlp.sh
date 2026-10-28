#!/bin/bash
# 19.10.2026 white-box-8c: fitmlp.py після V49DONE (сусід), без whest, 1 ядро. Кінець FITMLPDONE.
L=/home/trim/wblogs/v37_runs.log
until grep -q V49DONE $L; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
cd /home/trim/wb2lean/v39fija
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/trim/fs123/bin/python3 -I fitmlp.py > /home/trim/wblogs/fitmlp.log 2>&1
echo "FITMLPDONE rc=$?" >> $L
