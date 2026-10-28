#!/bin/bash
# 20.10.2026 white-box-89: orac_multi, процес на мережу, дві P104 паралельно (парні/непарні). Кінець ORACMULTI2DONE.
until grep -q ORACMULTIDONE /home/trim/wblogs/orac_multi.log; do sleep 20; done
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2
P=/home/trim/fs123/bin/python3.12; H=/home/trim/wb2lean/lt2re; O=/home/trim/wblogs
( for k in 0 2 4 6; do CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 $P $H/orac_multi2.py $k-$k $O/orac_multi2_a.json; done ) &
( for k in 1 3 5 7; do CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 $P $H/orac_multi2.py $k-$k $O/orac_multi2_b.json; done ) &
wait; echo ORACMULTI2DONE
