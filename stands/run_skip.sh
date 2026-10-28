#!/bin/bash
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 LEAN_DEV=cuda; shift
cd /home/trim/wb2lean; L=/home/trim/wblogs/lean_skip.log; PY=/home/trim/notqa-venv/bin/python3
for cfg in "$@"; do echo "== $cfg" >> $L; env $cfg $PY -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen rec 2>&1 | grep -v Warning | grep -E "MEAN|OLDSKIP: молодих" | sed "s/^/[$cfg] /" | sort -u | head -20 >> $L; done
echo "DONE_SKIP_$$" >> $L
