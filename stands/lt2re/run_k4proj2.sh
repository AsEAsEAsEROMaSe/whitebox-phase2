#!/bin/bash
# 20.10.2026 white-box-89: cov-проєкція K=4 на сідах 1..4 (як готова істина), r 16/32/64/128, A0 1/2, плюс none. Кінець K4PROJ2DONE.
export PYTHONPATH=/home/trim/arcpkg:/home/trim/ext/arc/mlp_cumulant_propagation/src:/home/trim/ext/fc/repo/tools CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef AW_DATA=/home/trim/wb2data/k4bed OMP_NUM_THREADS=2
P=/home/trim/notqa-venv/bin/python; S=/home/trim/wb2lean/lt2re/k4proj.py; O=/home/trim/wblogs/k4proj2.log
for s in 2 3 4 1; do
  until [ -f /home/trim/wb2data/k4bed/kscan_256_16_${s}_134000000.npz ]; do sleep 20; done
  [ $s != 1 ] && $P $S 256 16 $s 1.34e8 none 256 2 2>&1 | grep "k4proj\|Error" >> $O
  for r in 16 32 64 128; do $P $S 256 16 $s 1.34e8 cov $r 2 2>&1 | grep "k4proj\|Error" >> $O; done
  $P $S 256 16 $s 1.34e8 cov 64 1 2>&1 | grep "k4proj\|Error" >> $O
done
echo K4PROJ2DONE >> /home/trim/wblogs/v37_runs.log
