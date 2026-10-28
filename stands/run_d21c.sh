#!/bin/bash
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-bff69481-0d35-74f0-8871-68eb429cdc34 LEAN_DEV=cuda
cd /home/trim/wb2lean
L=/home/trim/wblogs/lean_d21c.log; echo "== start $(date)" >> $L
PY=/home/trim/notqa-venv/bin/python3
run() { tag=$1; md=$2; shift 2; echo "== $tag" >> $L; env "$@" $PY -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen $md 2>&1 | grep -v Warning | grep -E "dump|l= 1 D21|l= 8 D21|l=15 D21|MEAN" | cut -c1-220 >> $L; }
run proj_res64 rec D21USE=1 D21FEED=1 D21DIAG=0 D21PROJ=64 D21PBASIS=res
run proj_hisR16 rec D21USE=1 D21FEED=1 D21DIAG=0 D21PROJ=16 D21PBASIS=hisR
run proj_hisR64 rec D21USE=1 D21FEED=1 D21DIAG=0 D21PROJ=64 D21PBASIS=hisR
run proj_hisL64 rec D21USE=1 D21FEED=1 D21DIAG=0 D21PROJ=64 D21PBASIS=hisL
run proj_res64_k31 k31 D21USE=1 D21FEED=1 D21DIAG=0 D21PROJ=64 D21PBASIS=res
echo DONE_D21C >> $L
