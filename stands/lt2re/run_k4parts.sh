#!/bin/bash
# 20.10.2026 white-box-89: передатна функція κ4 по частинах читання (k4parts.py), мережа 1, кінець K4PARTSDONE
export PYTHONPATH=/home/trim/arcpkg:/home/trim/ext/arc/mlp_cumulant_propagation/src:/home/trim/ext/fc/repo/tools CUDA_DEVICE_ORDER=PCI_BUS_ID AW_DATA=/home/trim/wb2data/k4bed OMP_NUM_THREADS=2 AW_STORE=gpu:4 AW_SPEC=full
P=/home/trim/notqa-venv/bin/python; S=/home/trim/wb2lean/lt2re/k4parts.py; O=/home/trim/wblogs/k4parts.log
run() { CUDA_VISIBLE_DEVICES=$1 K4_PARTS=$2 $P $S 256 16 1 1.34e8 2>&1 | grep "k4parts\|Error\|Traceback" >> $O; }
a() { for p in 4,31,22,211 4 4,31 22,211; do run GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 $p; done; }
b() { for p in none 31 4,31,22 211; do run GPU-bff69481-0d35-74f0-8871-68eb429cdc34 $p; done; }
a & b & wait
echo K4PARTSDONE >> /home/trim/wblogs/v37_runs.log
