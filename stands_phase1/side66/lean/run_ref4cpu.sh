#!/bin/bash
# референс на CPU з вартовим за RSS (kill лише за явним PID дочірнього python)
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference OMP_NUM_THREADS=${THR:-4} MKL_NUM_THREADS=${THR:-4}
cd /home/trim/wb2lean; LOG=/home/trim/wblogs/ref4cpu_${TAG:-k3}.log
DEV=cpu F64=0 DUMPF=${DUMPF:-/home/trim/wb2data/dumps_p2/mlp_%04d.npz} python3 ref4.py ${DUMP:-0} ${K:-3} SIMPLE > $LOG 2>&1 &
PID=$!; MAXKB=${MAXKB:-14000000}; PEAK=0
while kill -0 $PID 2>/dev/null; do
  RSS=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null); [ -n "$RSS" ] && [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
  if [ -n "$RSS" ] && [ "$RSS" -gt "$MAXKB" ]; then echo "WATCHDOG: RSS $RSS kB > $MAXKB, kill $PID" >> $LOG; kill $PID; sleep 2; kill -9 $PID 2>/dev/null; fi
  sleep 5
done
echo "PEAK_RSS_KB $PEAK" >> $LOG; echo DONE_REF4CPU >> $LOG
