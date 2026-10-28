#!/bin/bash
# сторож RAM: вбиває МОЇ whest-прогони, якщо машина юзера почне задихатись
S0=$(free -m | awk '/Своп|Swap/{print $3}')
while pgrep -f "run_wall12.sh|run_sbr.sh|whest run" >/dev/null; do
  A=$(free -m | awk '/Пам|Mem/{print $7}'); S=$(free -m | awk '/Своп|Swap/{print $3}')
  if [ "$A" -lt 2000 ] || [ $((S - S0)) -gt 1500 ]; then
    pkill -f run_wall12.sh; pkill -f run_sbr.sh; pkill -f "whest run"
    echo "$(date +%H:%M:%S) ВБИТО: дост ${A} МБ, своп +$((S - S0)) МБ" >> /home/trim/wblogs/memguard.log; exit 0
  fi
  sleep 15
done
echo "$(date +%H:%M:%S) прогін завершився сам, сторож вийшов" >> /home/trim/wblogs/memguard.log
