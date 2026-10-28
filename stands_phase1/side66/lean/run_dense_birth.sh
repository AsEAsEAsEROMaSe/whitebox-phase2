#!/bin/bash
# GPU A: (1) dense-ланцюг з усіма членами: чи та сама похибка народження 3% T? (2) useall для дампів 4-7 + onestep SAVEP
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 LEAN_DEV=cuda
cd /home/trim/wb2lean; PY=/home/trim/notqa-venv/bin/python3; L=/home/trim/wblogs/dense_birth.log
echo "== dense chain dumps 0,3 $(date +%H:%M)" > $L
CFG=dense SAVE=/home/trim/wb2lean/his_dense_dump%d.npz $PY -W ignore d21_res.py 0,3 2>&1 | grep -E "^\[res" >> $L
for d in 0 3; do echo "== onestep dense dump $d $(date +%H:%M)" >> $L; HIS=/home/trim/wb2lean/his_dense_dump%d.npz $PY -W ignore onestep.py $d 2>&1 | grep -E "^l=(5|8|12|15) " | sed -E 's/.*(l=[ 0-9]+ ).*(РОЗКЛАД R: [^;]*).*(НАРОДЖЕННЯ Rb[^|]*).*/\1| \2 | \3/' >> $L; done
echo "== useall chain dumps 4-7 $(date +%H:%M)" >> $L
D21USE=1 D21FEED=1 K4FEED=1 K31FEED=1 TAG=useall $PY -W ignore birth_err.py 4,5,6,7 2>&1 | grep -E "^\[useall" >> $L
for d in 4 5 6 7; do echo "== onestep SAVEP dump $d $(date +%H:%M)" >> $L; HIS=/home/trim/wb2lean/his_useall_dump%d.npz SAVEP=/home/trim/wb2data/pieces_useall_%d.npz $PY -W ignore onestep.py $d 2>&1 | grep -E "saved pieces|^l=15 " | cut -c1-120 >> $L; done
echo DONE_DENSE_A >> $L
