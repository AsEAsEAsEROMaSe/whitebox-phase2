#!/bin/bash
# дешевший старий ярус через МЕНШЕ ПАР НІГ (не ранг): для джерел віку ≥A обнулити пари. Передбачення: X3/Y3 (κ₄-feed арми) з віку 4 коштує ≤3% MSE (шанс 50%); X2/Y2 (WK12/WK22) ≤5%; обидві ≤10% при ціні старого ярусу ÷2.
export PYTHONPATH=/home/trim/leanpkg:/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 LEAN_DEV=cuda
cd /home/trim/wb2lean; L=/home/trim/wblogs/lean_arm.log; PY=/home/trim/notqa-venv/bin/python3; shift
run() { tag=$1; shift; echo "== $tag" >> $L; env "$@" $PY -W ignore lean_oracle.py 0,1,2,3,4,5,6,7 regen rec 2>&1 | grep -v Warning | grep -E "MEAN" | sed "s/^/[$tag] /" >> $L; }
for cfg in "$@"; do run "$cfg" ARM_DROP=$cfg; done
echo "DONE_ARM_$$" >> $L
