#!/bin/bash
# 20.10.2026 white-box-89: другий прохід GPU-дампу corpus 2048..4095 (істина ext01..07 докачана white-box-16), 3 карти, крок 3.
# Чекає кінця першого проходу (GPUDUMPDONE і DONE gpudump rev). Кінець GPUDUMP2DONE.
L=/home/trim/wblogs/v37_runs.log; D=/home/trim/wb2data/lt2dump; H=/home/trim/wb2lean/lt2re
until grep -qx GPUDUMPDONE $L && grep -q "DONE gpudump rev" $L; do sleep 15; done
echo "START gpudump2 2048..4095 (white-box-89, 3 карти) $(date +%H:%M)" >> $L
run() {
  CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/trim/fs123/bin/python3.12 $H/gpu_dump.py corpus 2048-4095 $D/corpus --step 3 --off $2 >> /home/trim/wblogs/gpudump2_$2.log 2>&1
  echo "DONE gpudump2 off$2 rc=$? $(date +%H:%M)" >> $L
}
run GPU-bff69481-0d35-74f0-8871-68eb429cdc34 0 &
run GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 1 &
run GPU-ddffe75c-cebe-ec59-141d-3a922803464f 2 &
wait
echo GPUDUMP2DONE >> $L
