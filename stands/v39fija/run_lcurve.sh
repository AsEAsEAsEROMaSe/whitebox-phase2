#!/bin/bash
# 18.10.2026 white-box-21: крива навчання таблиці step_F87 (fitx.py, FIT_NUSE=25/35), 1 ядро, nice 19, ~360 МБ; таблиці в окремі файли.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/trim/wb2lean/v39fija
for k in 25 35; do
  FIT_NUSE=$k FIT_OUT=/home/trim/wb2lean/v39fija/fitx_tables_n$k.json nice -n 19 /home/trim/fs123/bin/python3 -I fitx.py > /home/trim/wblogs/fitx_n$k.log 2>&1
  echo "exit $?" >> /home/trim/wblogs/fitx_n$k.log
done
echo LCURVEDONE >> /home/trim/wblogs/fitx_n35.log
