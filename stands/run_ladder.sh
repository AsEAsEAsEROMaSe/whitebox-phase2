#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID LEAN_DEV=cuda OMP_NUM_THREADS=1
cd /home/trim/wb2lean
CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef DUMPS=0,3,6 nice -n 15 /home/trim/notqa-venv/bin/python3 -u -W ignore ladder.py 2>&1 | grep -v Warning > /home/trim/wblogs/ladder_a.log &
CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 DUMPS=1,4,7 nice -n 15 /home/trim/notqa-venv/bin/python3 -u -W ignore ladder.py 2>&1 | grep -v Warning > /home/trim/wblogs/ladder_b.log &
CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26 DUMPS=2,5 nice -n 15 /home/trim/notqa-venv/bin/python3 -u -W ignore ladder.py 2>&1 | grep -v Warning > /home/trim/wblogs/ladder_c.log &
wait; echo ALLDONE_LADDER >> /home/trim/wblogs/ladder_a.log
