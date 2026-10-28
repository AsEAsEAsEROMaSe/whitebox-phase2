#!/bin/bash
# 20.10.2026 white-box-89: orac_c2 (λ і ранг) після orac_c, мережі 0..5 на трьох P104 (не 1080). Кінець ORACC2DONE.
O=/home/trim/wblogs
while pgrep -f "lt2re/orac_c.py" >/dev/null; do sleep 20; done
export NMC=33554432 CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2
P=/home/trim/fs123/bin/python3.12; S=/home/trim/wb2lean/lt2re/orac_c2.py
CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 $P $S 0-1 $O/orac_c2_a.json > $O/orac_c2_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 $P $S 2-3 $O/orac_c2_b.json > $O/orac_c2_b.log 2>&1 &
CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef $P $S 4-5 $O/orac_c2_c.json > $O/orac_c2_c.log 2>&1 &
wait
echo ORACC2DONE >> $O/v37_runs.log
