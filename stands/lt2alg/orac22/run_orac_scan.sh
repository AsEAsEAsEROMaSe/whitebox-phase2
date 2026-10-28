#!/bin/bash
# 22.10.2026 b9: оракул зрізів κ4 у ланцюгу 337568, мережі 0,2,1, карта 26c6ed1b (дозвіл юзера через c2)
cd /home/trim/wb2lean/lt2alg/orac22
env CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 \
  /home/trim/fs123/bin/python3.12 run_orac_scan.py 0,2,1 /home/trim/wblogs/orac22_scan.json
