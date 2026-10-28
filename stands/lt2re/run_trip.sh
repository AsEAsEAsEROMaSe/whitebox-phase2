#!/bin/bash
# 21.10.2026 white-box-f9: triple-частина D3 від обрізання K3, конфіги поза бюджетом. Кінець TRIPDONE.
H=/home/trim/wb2lean/lt2re; O=/home/trim/wblogs/trip_rank.jsonl
run() { local tag=$1; shift
  env "$@" CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    timeout 1200 /home/trim/fs123/bin/python3.12 $H/trip_rank.py 1 $tag $O > /home/trim/wblogs/trip_$tag.log 2>&1 || echo "FAIL $tag" >> /home/trim/wblogs/trip_fail.log; }
run base X=0
run yeps0 AW_Y_EPS=0
run noconf V21_NO_CONFINE=1
run rold1024 V21_R_OLD=1024 V24_AGE_OLD2=0
run age15 V21_AGE_OLD=15
run nofija SB1_FIJA=0
run rfb32 AW_R_FB=32
run noprune V30_PRUNE=0
run fulllast V19_FULL_LAST=1
run all AW_Y_EPS=0 V21_NO_CONFINE=1 V30_PRUNE=0 AW_R_FB=32
echo "TRIPDONE (white-box-f9) $(date +%H:%M)" >> /home/trim/wblogs/v37_runs.log
