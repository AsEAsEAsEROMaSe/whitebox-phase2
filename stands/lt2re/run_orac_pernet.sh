#!/bin/bash
# 20.10.2026 white-box-89: по одному процесу на мережу (буфери ланцюга не звільняються між мережами ⇒ OOM). Кінець ORACPERNETDONE.
O=/home/trim/wblogs
while pgrep -f "lt2re/orac_c2.py" >/dev/null; do sleep 20; done
export NMC=33554432 CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2
P=/home/trim/fs123/bin/python3.12; H=/home/trim/wb2lean/lt2re
lane() { local G=$1; shift; export CUDA_VISIBLE_DEVICES=$G; for job in "$@"; do s=${job%%:*}; k=${job##*:}; $P $H/$s.py $k-$k $O/${s}_n$k.json >> $O/${s}_pernet.log 2>&1; done; }
lane GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 orac_c2:1 orac_c:2 orac_c:3 orac_c:8 orac_c:9 &
lane GPU-bff69481-0d35-74f0-8871-68eb429cdc34 orac_c2:3 orac_c:4 orac_c:5 orac_c:10 orac_c:11 &
lane GPU-eaa8132f-e46f-2090-d943-0481fbe03aef orac_c2:5 orac_c:14 orac_c:15 orac_c:16 orac_c:17 &
wait
echo ORACPERNETDONE >> $O/v37_runs.log
