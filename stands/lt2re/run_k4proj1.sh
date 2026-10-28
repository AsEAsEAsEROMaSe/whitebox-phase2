#!/bin/bash
# 20.10.2026 white-box-89: k4proj на 256×16 seed 1: rand/adj/cov при r 32 і 64, A0=2. Кінець K4PROJ1DONE.
export PYTHONPATH=/home/trim/arcpkg:/home/trim/ext/arc/mlp_cumulant_propagation/src:/home/trim/ext/fc/repo/tools CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef AW_DATA=/home/trim/wb2data/k4bed OMP_NUM_THREADS=2
for r in 32 64; do for m in adj rand cov; do
  /home/trim/notqa-venv/bin/python /home/trim/wb2lean/lt2re/k4proj.py 256 16 1 1.34e8 $m $r 2 2>&1 | grep "k4proj\|Error\|Traceback" >> /home/trim/wblogs/k4proj1.log
done; done
echo K4PROJ1DONE >> /home/trim/wblogs/v37_runs.log
