#!/bin/bash
# 20.10.2026 white-box-89: стиснення МОЛОДИХ груп K=4 одразу при народженні (A0=0) рангом r = 128/192 (n/2, 3n/4), сіди 1,2. Кінець K4PROJ3DONE.
export PYTHONPATH=/home/trim/arcpkg:/home/trim/ext/arc/mlp_cumulant_propagation/src:/home/trim/ext/fc/repo/tools CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef AW_DATA=/home/trim/wb2data/k4bed OMP_NUM_THREADS=2
P=/home/trim/notqa-venv/bin/python; S=/home/trim/wb2lean/lt2re/k4proj.py; O=/home/trim/wblogs/k4proj3.log
for s in 1 2; do for r in 128 192; do $P $S 256 16 $s 1.34e8 cov $r 0 2>&1 | grep "k4proj\|Error" >> $O; done; done
echo K4PROJ3DONE >> /home/trim/wblogs/v37_runs.log
