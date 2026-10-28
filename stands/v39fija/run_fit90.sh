#!/bin/bash
# 18.10.2026 white-box-21: після V43LNDONE (дамп ⟹ a5 v44 ⟹ v43ln) і 60 с без whest ⟹ wextract.py WX_ALL=1 (ваги нових 40, повтори, якщо з'явиться whest)
# ⟹ WEXTRACT2DONE ⟹ fit90.py (лог fit90.log, 1 ядро, nice 19) ⟹ FIT90DONE.
L=/home/trim/wblogs/v37_runs.log
PY=/home/trim/fs123/bin/python3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
idle60() { for i in 1 2 3 4 5 6; do pgrep -f "bin/whest run" >/dev/null && return 1; sleep 10; done; return 0; }
until grep -q '^V43LNDONE' $L; do sleep 30; done   # після черги a5 (v44) і мого v43ln
for try in $(seq 1 20); do
  until idle60; do sleep 20; done
  WX_ALL=1 nice -n 19 $PY -I /home/trim/wb2lean/v39fija/wextract.py >> /home/trim/wblogs/wextract2.log 2>&1
  echo "try $try exit $?" >> /home/trim/wblogs/wextract2.log
  grep -q '^WEXTRACT2DONE' $L && break
done
grep -q '^WEXTRACT2DONE' $L || { echo "WEXTRACT2 FAILED" >> $L; exit 1; }
cd /home/trim/wb2lean/v39fija
nice -n 19 $PY -I fit90.py > /home/trim/wblogs/fit90.log 2>&1; echo "exit $?" >> /home/trim/wblogs/fit90.log
echo "FIT90DONE" >> $L
