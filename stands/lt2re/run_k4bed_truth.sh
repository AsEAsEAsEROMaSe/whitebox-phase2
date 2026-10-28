#!/bin/bash
# 20.10.2026 white-box-89: стенд K=4 на ширині 256×16: істина МК (kscan.py Susupro, MIT) + k3a, сіди 1..4. Кінець K4BEDTRUTHDONE.
L=/home/trim/wblogs/v37_runs.log
echo "START k4bed truth (white-box-89, GPU eaa8132f) $(date +%H:%M)" >> $L
export PYTHONPATH=/home/trim/arcpkg:/home/trim/ext/arc/mlp_cumulant_propagation/src:/home/trim/ext/fc/repo/tools
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef AW_DATA=/home/trim/wb2data/k4bed AW_THREADS=2 AW_KDEV=cuda
cd /home/trim/ext/fc/repo/lab
for s in 1 2 3 4; do
  /home/trim/notqa-venv/bin/python kscan.py 256 16 $s 1.34e8 k3a >> /home/trim/wblogs/k4bed_truth.log 2>&1
done
echo K4BEDTRUTHDONE >> $L
