#!/bin/bash
# 18.10.2026 a5 v46 (точні заощадження e30_a ueye/tri/nestg і gsum поверх v45 з рангами 4/8): після V45DONE і без whest, MemAvailable >= 14 ГБ.
# Пломба v46ueye_s0 (лише UEYE: мусить дорівнювати v45_s0 біт у біт), далі v46 s0..s9 проти v45. Кінець V46DONE.
L=/home/trim/wblogs/v37_runs.log
until grep -qE '^(RA32DONE|STOP ra32)' $L; do sleep 30; done   # після тесту ROWALIGN 32 white-box-21 (він чекає V45DONE)
until grep -q '^PACK v45 rc=' $L; do sleep 20; done   # окремий пакувальник queue_pack45.sh (дозвіл юзера на сабміт v45)
sleep 90   # white-box-21: якщо юзер дав дозвіл на v45a (арени), вона пише V45A QUEUED, і v46 чекає її кінця
if grep -q '^V45A QUEUED' $L; then until grep -qE '^(V45ADONE|STOP v45a)' $L; do sleep 30; done; fi
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 14 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
E=/home/trim/wb2lean/v39fija/estimator_v46.py
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh $E v46ueye_s0 1 V46_TRI=0 V46_NESTG=0 V46_GSUM=0
for s in s0 s1 s2 s3 s4 s5 s6 s7 s8 s9; do DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v46_$s 1; done
echo V46DONE >> $L
