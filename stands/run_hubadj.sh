#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_PATH=/usr CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-eaa8132f-e46f-2090-d943-0481fbe03aef LEAN_DEV=cuda OMP_NUM_THREADS=1
cd /home/trim/wb2lean
for job in "adj 0" "fd 0" "adj 1"; do
  set -- $job
  echo "START $1 dump $2 $(date +%T)"
  MODE=$1 DUMP=$2 L0=11 FD_LAYERS=3,6,12 NCOL=48 nice -n 15 /home/trim/notqa-venv/bin/python3 -u -W ignore hubadj.py 2>&1 | grep -v Warning
  echo "END $1 dump $2 $(date +%T)"
done
echo ALLDONE_HUBADJ
