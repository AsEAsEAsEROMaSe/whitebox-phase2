#!/bin/bash
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 LEAN_DEV=cuda OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd /home/trim/wb2lean; shift
for i in "$@"; do /home/trim/notqa-venv/bin/python3 -W ignore -u knobs_t.py $i >> /home/trim/wblogs/knt.log 2>&1; done
echo DONE_ALL $@ >> /home/trim/wblogs/knt.log
