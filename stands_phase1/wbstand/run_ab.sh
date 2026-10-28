#!/bin/bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
cd /mnt/win_d/llm/White-Box-P2
DS=/home/trim/wb2data/dev24
B=2199023255552          # 2^41 — бюджет ФАЗИ 2
for E in estimator_p2_BEST_v11 estimator_p2_a163; do
  echo "=== $E (flop-budget $B) ==="
  /home/trim/fsnew/bin/whest run --estimator $E.py --dataset $DS --split mini \
      --flop-budget $B --format json 2>/dev/null > /home/trim/wblogs/ab2_$E.json
done
echo "ГОТОВО_AB2"
