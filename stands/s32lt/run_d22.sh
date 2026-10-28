#!/bin/bash
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 OMP_NUM_THREADS=1
P=/home/trim/fs123/bin/python3.12; O=/home/trim/wblogs/d22; mkdir -p $O
$P t_d22.py 0-2 $O/base.json > $O/base.log 2>&1
X_D22_LOG=1 X_D22_B=0 $P t_d22.py 0-2 $O/b000.json > $O/b000.log 2>&1
for b in 0.5 0.35 0.7 1.0; do X_D22_B=$b $P t_d22.py 0-2 $O/b$b.json > $O/b$b.log 2>&1; done
echo D22DONE >> $O/base.log
