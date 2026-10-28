#!/bin/bash
# 21.10.2026 white-box-b9: дамп ознак s32lt на корпусі 0..999, дві карти (P104 eaa8132f + 1080 Ti ddffe75c), кінець ALGDUMPDONE.
L=/home/trim/wblogs/v37_runs.log; D=/home/trim/wb2data/lt2alg_dump; H=/home/trim/wb2lean/lt2alg
echo "START algdump (white-box-b9, eaa8132f + ddffe75c) $(date +%H:%M)" >> $L
run() {
  export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  /home/trim/fs123/bin/python3.12 $H/gpu_dump_alg.py corpus 0-999 $D --step 2 --off $2 >> /home/trim/wblogs/algdump_$2.log 2>&1
  echo "DONE algdump off$2 rc=$? $(date +%H:%M)" >> $L
}
run GPU-eaa8132f-e46f-2090-d943-0481fbe03aef 0 &
run GPU-ddffe75c-cebe-ec59-141d-3a922803464f 1 &
wait
echo ALGDUMPDONE >> $L
