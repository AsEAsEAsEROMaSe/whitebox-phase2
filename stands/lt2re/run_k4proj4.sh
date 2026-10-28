#!/bin/bash
# 20.10.2026 white-box-89: «кубик Рубика» — стиснення груп K=4 у власному базисі (HOSVD першої розгортки), сід 1:
# A0=0 (молоді одразу) r 64/128, A0=2 r 64 (порівняти з cov 2.19e-7). Кінець K4PROJ4DONE.
export PYTHONPATH=/home/trim/arcpkg:/home/trim/ext/arc/mlp_cumulant_propagation/src:/home/trim/ext/fc/repo/tools CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 AW_DATA=/home/trim/wb2data/k4bed OMP_NUM_THREADS=2
P=/home/trim/notqa-venv/bin/python; S=/home/trim/wb2lean/lt2re/k4proj.py; O=/home/trim/wblogs/k4proj4.log
until grep -qx GPUDUMP2DONE /home/trim/wblogs/v37_runs.log; do sleep 15; done
for a in "64 0" "128 0" "64 2"; do set -- $a; $P $S 256 16 1 1.34e8 hosvd $1 $2 2>&1 | grep "k4proj\|Error\|Traceback" >> $O; done
echo K4PROJ4DONE >> /home/trim/wblogs/v37_runs.log
