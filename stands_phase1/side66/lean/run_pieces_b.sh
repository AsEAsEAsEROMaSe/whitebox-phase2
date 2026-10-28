#!/bin/bash
# GPU B: useall для дампів 1,2 + onestep SAVEP для дампів 0-3
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda
cd /home/trim/wb2lean; PY=/home/trim/notqa-venv/bin/python3; L=/home/trim/wblogs/pieces_b.log
echo "== useall chain dumps 1,2 $(date +%H:%M)" > $L
D21USE=1 D21FEED=1 K4FEED=1 K31FEED=1 TAG=useall $PY -W ignore birth_err.py 1,2 2>&1 | grep -E "^\[useall" >> $L
for d in 0 1 2 3; do echo "== onestep SAVEP dump $d $(date +%H:%M)" >> $L; HIS=/home/trim/wb2lean/his_useall_dump%d.npz SAVEP=/home/trim/wb2data/pieces_useall_%d.npz $PY -W ignore onestep.py $d 2>&1 | grep -E "saved pieces|^l=15 " | cut -c1-120 >> $L; done
echo DONE_PIECES_B >> $L
