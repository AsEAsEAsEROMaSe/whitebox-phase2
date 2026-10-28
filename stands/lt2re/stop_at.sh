#!/bin/bash
# 20.10.2026: о 21:55 зупинити все моє навантаження (прохання юзера)
T=$(date -d "21:55" +%s); N=$(date +%s); [ $T -gt $N ] && sleep $((T-N))
for pat in 'run_orac' 'run_chres' 'run_breaks' 'lt2re/.*\.py'; do pkill -f "$pat"; done
sleep 5; for pat in 'lt2re/.*\.py'; do pkill -9 -f "$pat"; done
date >> /home/trim/wblogs/stop_at.log; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> /home/trim/wblogs/stop_at.log
~/.claude/hooks/say "Двадцять перша п'ятдесят п'ять, усе навантаження вимкнено."
