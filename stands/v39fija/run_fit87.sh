#!/bin/bash
# 18.10.2026 a2: чекає ПОВНИЙ дамп a5 (60 файлів, непорожній json, жоден whest не йде), тоді фіти E-A (CPU numpy, легкі)
D=/home/trim/wblogs/corrdump; J=/home/trim/wblogs/v37_corrdump60b.json
while true; do
  n=$(ls $D/net_*.npz 2>/dev/null | wc -l)
  if [ "$n" -ge 60 ] && [ -s "$J" ] && ! pgrep -f "whest run" >/dev/null; then break; fi
  sleep 60
done
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/trim/fs123/bin/python3
$PY /home/trim/wb2lean/v39fija/fit87.py $D $J > /home/trim/wblogs/fit87.log 2>&1
$PY /home/trim/wb2lean/v39fija/fit87b.py $D $J > /home/trim/wblogs/fit87b.log 2>&1
echo FIT87DONE >> /home/trim/wblogs/fit87b.log
