#!/bin/bash
# 20.10.2026 white-box-89: «розширення» — спектр ядра кожного атома новонароджених груп K=4 і обрізання атома до рангу k при народженні.
# AW_SPEC розділяє гаусові (path+star), C·κ3 (three2) і зрізові групи. Сід 1. Кінець K4ATOMDONE.
# Передбачення: гаусові атоми ранг ~1-2 (energy@2 > 0.99), three2 і зрізи повільніше; k=16 для всіх тримає >= 80% log-виграшу; шанс 30%.
export PYTHONPATH=/home/trim/arcpkg:/home/trim/ext/arc/mlp_cumulant_propagation/src:/home/trim/ext/fc/repo/tools CUDA_DEVICE_ORDER=PCI_BUS_ID AW_DATA=/home/trim/wb2data/k4bed OMP_NUM_THREADS=2
export AW_SPEC="path+star,three2,slice4+slice31+slice22+slice211"
P=/home/trim/notqa-venv/bin/python; S=/home/trim/wb2lean/lt2re/k4proj.py; O=/home/trim/wblogs/k4atom.log
a() { CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 K4_ATOM_SPEC=1 $P $S 256 16 1 1.34e8 none 256 2 2>&1 | grep "k4proj\|atoms\|Error\|Traceback" >> $O.a
       CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 K4_ATOM_RANK=16 $P $S 256 16 1 1.34e8 none 256 2 2>&1 | grep "k4proj\|Error\|Traceback" >> $O.a; }
b() { CUDA_VISIBLE_DEVICES=GPU-ddffe75c-cebe-ec59-141d-3a922803464f K4_ATOM_RANK=4 $P $S 256 16 1 1.34e8 none 256 2 2>&1 | grep "k4proj\|Error\|Traceback" >> $O.b
       CUDA_VISIBLE_DEVICES=GPU-ddffe75c-cebe-ec59-141d-3a922803464f K4_ATOM_RANK=1 $P $S 256 16 1 1.34e8 none 256 2 2>&1 | grep "k4proj\|Error\|Traceback" >> $O.b; }
a & b & wait
echo K4ATOMDONE >> /home/trim/wblogs/v37_runs.log
