#!/bin/bash
# 20.10.2026 white-box-89: orac_crit (критичні нейрони), мережі 0..5, процес на мережу, після ORACPERNETDONE. Кінець ORACCRITDONE.
O=/home/trim/wblogs
until grep -qx ORACPERNETDONE $O/v37_runs.log; do sleep 20; done
export NMC=33554432 CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2
P=/home/trim/fs123/bin/python3.12; S=/home/trim/wb2lean/lt2re/orac_crit.py
lane() { export CUDA_VISIBLE_DEVICES=$1; shift; for k in "$@"; do $P $S $k-$k $O/orac_crit_n$k.json >> $O/orac_crit.log 2>&1; done; }
lane GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 0 3 &
lane GPU-bff69481-0d35-74f0-8871-68eb429cdc34 1 4 &
lane GPU-eaa8132f-e46f-2090-d943-0481fbe03aef 2 5 &
wait
echo ORACCRITDONE >> $O/v37_runs.log
