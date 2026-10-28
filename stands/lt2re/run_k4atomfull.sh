#!/bin/bash
# 20.10.2026 white-box-89: спектр атомів ЗЛИТОЇ двоногої групи (path+star+three2 в одному Bb, де пари вже погашені). Кінець K4ATOMFULLDONE.
# Передбачення: злита група низькорангова сильніше за окремі (energy@16 > 0.95 на ранніх шарах); шанс 25%.
export PYTHONPATH=/home/trim/arcpkg:/home/trim/ext/arc/mlp_cumulant_propagation/src:/home/trim/ext/fc/repo/tools CUDA_DEVICE_ORDER=PCI_BUS_ID AW_DATA=/home/trim/wb2data/k4bed OMP_NUM_THREADS=2 AW_SPEC=full
CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 K4_ATOM_SPEC=1 /home/trim/notqa-venv/bin/python /home/trim/wb2lean/lt2re/k4proj.py 256 16 1 1.34e8 none 256 2 > /home/trim/wblogs/k4atomfull.log 2>&1
echo K4ATOMFULLDONE >> /home/trim/wblogs/v37_runs.log
