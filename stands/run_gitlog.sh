#!/bin/bash
# git-лог ланцюга: дампи по черзі, один процес, карта eaa8 (P104), nice 15
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/trim/wb2lean
for i in "$@"; do nice -n 15 /home/trim/notqa-venv/bin/python3 gitlog.py $i 2>&1 | grep -v Warning > /home/trim/wblogs/gitlog_$i.log; done
echo ALL_DONE >> /home/trim/wblogs/gitlog_${!#}.log
