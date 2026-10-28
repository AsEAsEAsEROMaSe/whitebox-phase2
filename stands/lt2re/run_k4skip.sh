#!/bin/bash
# 20.10.2026 white-box-89: «мовчазні молоді» — читання груп K=4 віком <= S вимкнено (стан переноситься), S=1,2; і S=1 + cov r64 A0=2.
# Сіди 1 (на 26c6ed1b) і 2 (на 1080 Ti) паралельно. Кінець K4SKIPDONE.
# Передбачення: S=1 тримає >= 50% log-виграшу K=4 над k3a; S=2 < 30%; шанс 20%.
export PYTHONPATH=/home/trim/arcpkg:/home/trim/ext/arc/mlp_cumulant_propagation/src:/home/trim/ext/fc/repo/tools CUDA_DEVICE_ORDER=PCI_BUS_ID AW_DATA=/home/trim/wb2data/k4bed OMP_NUM_THREADS=2
P=/home/trim/notqa-venv/bin/python; S=/home/trim/wb2lean/lt2re/k4proj.py; O=/home/trim/wblogs/k4skip.log
one() {  # seed gpu
  export CUDA_VISIBLE_DEVICES=$2
  K4_SKIP_YOUNG_READS=1 $P $S 256 16 $1 1.34e8 none 256 2 2>&1 | grep "k4proj\|Error\|Traceback" >> $O
  K4_SKIP_YOUNG_READS=2 $P $S 256 16 $1 1.34e8 none 256 2 2>&1 | grep "k4proj\|Error\|Traceback" >> $O
  K4_SKIP_YOUNG_READS=1 $P $S 256 16 $1 1.34e8 cov 64 2 2>&1 | grep "k4proj\|Error\|Traceback" >> $O
}
one 1 GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 &
one 2 GPU-ddffe75c-cebe-ec59-141d-3a922803464f &
wait
echo K4SKIPDONE >> /home/trim/wblogs/v37_runs.log
