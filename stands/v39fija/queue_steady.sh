#!/bin/bash
# 18.10.2026 white-box-21: сталий стан грейдера (кілька мереж в одному процесі: пули/види вже є, Штрассен L5).
# v45 і v45b на парах p2 і p4 (по 2 мережі за прогін); друга мережа кожного прогону = сталий стан. Після V46DONE (a5). Кінець STEADYDONE.
L=/home/trim/wblogs/v37_runs.log
until grep -qE '^(V46DONE|STOP v46)' $L; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 14 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
for p in p2 p4; do
  DSX=/home/trim/wb2data/dsub/$p ./run2.sh /home/trim/wb2lean/v39fija/estimator_v45.py st45_$p 2
  DSX=/home/trim/wb2data/dsub/$p ./run2.sh /home/trim/wb2lean/v39fija/estimator_v45b.py st45b_$p 2
done
echo STEADYDONE >> $L
