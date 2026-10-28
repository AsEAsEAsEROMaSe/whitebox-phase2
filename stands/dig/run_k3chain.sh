#!/bin/bash
# 16.10.2026 вечір: предиктор κ4 з κ3 власних джерел ланцюга, 8 дампів, далі фіт і ланцюг k4selp sens 25%. Лише GPU eaa8, один процес.
cd /home/trim/wb2lean
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
P=/home/trim/notqa-venv/bin/python3
echo "== $(date)"
for i in ${DUMPS:-0 1 2 3 4 5 6 7}; do nice -n 15 $P -W ignore dig/k4pred_k3chain.py $i 2>&1 | grep -v "^\s*$"; done
[ -n "$ONLY_RAW" ] && { echo ALLDONE; exit 0; }
nice -n 15 $P -W ignore dig/k4pred_fit.py k3chain_ 2>&1
echo "ланцюг k4selp sens 25%: $(SEL=sens FRAC=0.25 K4PSUF=k3chain_ nice -n 15 $P -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen k4selp 2>&1 | grep -E 'MEAN mse_final|Error' | tail -1)"
echo ALLDONE
