#!/bin/bash
# 18.10.2026 a2: чекає DUMP3DONE (дамп a5 по одній мережі, m10..m59) і відсутність whest, тоді fit87c.py (CPU numpy, пік ~1–2 ГБ);
# по завершенні (і при падінні) дописує FIT87CDONE у v37_runs.log, на нього чекає черга a5
while true; do
  if grep -q '^DUMP3DONE' /home/trim/wblogs/v37_runs.log && ! pgrep -f "bin/whest run" >/dev/null; then break; fi
  sleep 30
done
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/home/trim/fs123/bin/python3 -I /home/trim/wb2lean/v39fija/fit87c.py 50 > /home/trim/wblogs/fit87c.log 2>&1
echo "FIT87CDONE rc=$?" >> /home/trim/wblogs/fit87c.log
echo FIT87CDONE >> /home/trim/wblogs/v37_runs.log
