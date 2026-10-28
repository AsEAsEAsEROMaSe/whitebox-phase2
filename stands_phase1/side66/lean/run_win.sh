#!/bin/bash
# $1 = GPU uuid, $2 = тег лога, далі трійки tag:LMIN:LMAX
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 LEAN_DEV=cuda OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd /home/trim/wb2lean; L=/home/trim/wblogs/win_$2.log; echo "== start $(date)" >> $L; shift 2
PY=/home/trim/notqa-venv/bin/python3
for t in "$@"; do IFS=: read tag a b <<< "$t"
  if [ "$tag" = audit ]; then U=0; else U=1; fi
  echo "== $tag $a..$b" >> $L
  env D21USE=$U D21LMIN=$a D21LMAX=$b $PY -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen rec 2>&1 | grep -v Warning | grep -E "dump|MEAN" >> $L
done; echo DONE >> $L
