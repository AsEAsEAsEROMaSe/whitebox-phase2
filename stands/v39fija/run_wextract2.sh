#!/bin/bash
# 18.10.2026 white-box-21 (a2): чекає V40DONE від a5 (кінець queue_v40c) і 60 с без whest; тоді wextract.py.
# Якщо wextract зупинився через новий whest, чекає знову (до 20 спроб). Після WEXTRACTDONE: fitj.py, далі fitx.py, лише коли whest нема; 1 ядро, nice 19.
L=/home/trim/wblogs/v37_runs.log
PY=/home/trim/fs123/bin/python3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
idle60() { for i in 1 2 3 4 5 6; do pgrep -f "bin/whest run" >/dev/null && return 1; sleep 10; done; return 0; }
v40end() { grep -q '^V40DONE' $L; }
until v40end; do sleep 20; done
for try in $(seq 1 20); do
  until idle60; do sleep 20; done
  nice -n 19 $PY -I /home/trim/wb2lean/v39fija/wextract.py >> /home/trim/wblogs/wextract.log 2>&1
  echo "try $try exit $?" >> /home/trim/wblogs/wextract.log
  grep -q '^WEXTRACTDONE' $L && break
done
grep -q '^WEXTRACTDONE' $L || { echo "WEXTRACT FAILED" >> $L; exit 1; }
cd /home/trim/wb2lean/v39fija
until idle60; do sleep 20; done
nice -n 19 $PY -I fitj.py > /home/trim/wblogs/fitj.log 2>&1; echo "exit $?" >> /home/trim/wblogs/fitj.log
echo "FITJDONE" >> $L
until idle60; do sleep 20; done
nice -n 19 $PY -I fitx.py > /home/trim/wblogs/fitx.log 2>&1; echo "exit $?" >> /home/trim/wblogs/fitx.log
echo "FITXDONE" >> $L
