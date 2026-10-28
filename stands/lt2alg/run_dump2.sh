#!/bin/bash
# 21.10.2026 white-box-b9: дамп ознак s32lt (з LT_FE3 від a0, X 53 колонки) на корпусі 1000..2999, чотири карти (дозвіл юзера),
# вихід на /mnt/gguf (корінь повний), кінець ALGDUMP2DONE.
L=/home/trim/wblogs/v37_runs.log; D=/mnt/gguf/wb_lt2alg/dump2; H=/home/trim/wb2lean/lt2alg
echo "START algdump2 (white-box-b9, 4 карти) $(date +%H:%M)" >> $L
run() {
  export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  /home/trim/fs123/bin/python3.12 $H/gpu_dump_alg.py corpus 1000-2999 $D --step 4 --off $2 >> /home/trim/wblogs/algdump2_$2.log 2>&1
  echo "DONE algdump2 off$2 rc=$? $(date +%H:%M)" >> $L
}
run GPU-eaa8132f-e46f-2090-d943-0481fbe03aef 0 &
run GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 1 &
run GPU-bff69481-0d35-74f0-8871-68eb429cdc34 2 &
run GPU-ddffe75c-cebe-ec59-141d-3a922803464f 3 &
wait
echo ALGDUMP2DONE >> $L
