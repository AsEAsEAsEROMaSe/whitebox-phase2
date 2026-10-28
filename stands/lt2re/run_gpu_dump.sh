#!/bin/bash
# 19.10.2026 white-box-89: GPU-дамп ознак LT2 (gpu_dump.py) на двох P104: mini 0..99, далі corpus 0..4095, у теки white-box-16. Кінець GPUDUMPDONE.
L=/home/trim/wblogs/v37_runs.log; D=/home/trim/wb2data/lt2dump; H=/home/trim/wb2lean/lt2re
echo "START gpudump (white-box-89, P104 bff69481 + 26c6ed1b, по 1 ядру) $(date +%H:%M)" >> $L
run() {  # $1 uuid, $2 off
  export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  /home/trim/fs123/bin/python3.12 $H/gpu_dump.py mini 0-99 $D/mini --step 2 --off $2 >> /home/trim/wblogs/gpudump_$2.log 2>&1
  /home/trim/fs123/bin/python3.12 $H/gpu_dump.py corpus 0-4095 $D/corpus --step 2 --off $2 >> /home/trim/wblogs/gpudump_$2.log 2>&1
  echo "DONE gpudump off$2 rc=$? $(date +%H:%M)" >> $L
}
run GPU-bff69481-0d35-74f0-8871-68eb429cdc34 0 &
run GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 1 &
wait
echo GPUDUMPDONE >> $L
