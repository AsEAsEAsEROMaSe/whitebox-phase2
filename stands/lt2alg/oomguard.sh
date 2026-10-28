#!/bin/bash
# 21.10.2026 white-box-b9: OOMGUARD: MemAvailable < 4 ГБ ⟹ вбити найменш важливий мій прогін (D, далі C), до 01:30.
while [ $(date +%H%M) != 0130 ]; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ $a -lt 4000000 ]; then
    p=$(pgrep -f "lt2_train_seed.py /mnt/gguf/wb_lt2alg/prepB runs/[m]idD" || pgrep -f "lt2_train_seed.py /mnt/gguf/wb_lt2alg/prepA runs/[m]idB1")
    [ -n "$p" ] && kill $p && echo "$(date +%H:%M) OOMGUARD killed $p avail $a" >> /home/trim/wblogs/v37_runs.log
  fi
  sleep 10
done
