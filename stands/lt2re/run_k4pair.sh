#!/bin/bash
# 20.10.2026 white-box-89: гасіння пар діаграм K=4 (k4pair.py), мережа 1, кінець K4PAIRDONE
export PYTHONPATH=/home/trim/arcpkg:/home/trim/ext/arc/mlp_cumulant_propagation/src:/home/trim/ext/fc/repo/tools CUDA_DEVICE_ORDER=PCI_BUS_ID AW_DATA=/home/trim/wb2data/k4bed OMP_NUM_THREADS=2 AW_STORE=gpu:4
CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 /home/trim/notqa-venv/bin/python /home/trim/wb2lean/lt2re/k4pair.py 256 16 1 1.34e8 /home/trim/wblogs/k4pair_s1.npz > /home/trim/wblogs/k4pair.log 2>&1
echo K4PAIRDONE >> /home/trim/wblogs/v37_runs.log
