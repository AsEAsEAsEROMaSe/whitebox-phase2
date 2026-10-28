#!/bin/bash
# 18.10.2026 a2: дамп ознак v39 (без корекції, як corrdump a5) на НОВИХ мережах mini 60..99 по одній
# (набори dsub/m60..m99 зробив wextract.py) ⟹ дофіт таблиці на 90 мережах. Окремий каталог corrdump2.
# Чекає маркера, переданого першим аргументом (узгоджений з a5), і відсутності whest; кінець → DUMP4DONE.
M=${1:-WEXTRACTDONE}
while true; do
  if grep -q "^$M" /home/trim/wblogs/v37_runs.log && ! pgrep -f "bin/whest run" >/dev/null; then break; fi
  sleep 30
done
mkdir -p /home/trim/wblogs/corrdump2
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
for i in $(seq 60 99); do
  [ -s /home/trim/wblogs/v37_cd_m$i.json ] && continue
  DSX=/home/trim/wb2data/dsub/m$i ./run2.sh /home/trim/wb2lean/v39fija/estimator_v39_dump.py cd_m$i 1 V39_CORRDUMP=/home/trim/wblogs/corrdump2
done
echo DUMP4DONE >> /home/trim/wblogs/v37_runs.log
