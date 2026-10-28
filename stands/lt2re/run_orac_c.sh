#!/bin/bash
# 20.10.2026 white-box-89: оракул парного стану (orac_c.py), мережі корпусу 0..11 на двох картах. Кінець ORACCDONE.
export NMC=33554432 CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2
P=/home/trim/fs123/bin/python3.12; S=/home/trim/wb2lean/lt2re/orac_c.py; O=/home/trim/wblogs
CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 $P $S ${1:-0-5} $O/orac_c_a.json > $O/orac_c_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 $P $S ${2:-6-11} $O/orac_c_b.json > $O/orac_c_b.log 2>&1 &
wait
echo ORACCDONE >> $O/v37_runs.log
