#!/bin/bash
while ! grep -q DONE_MISS3 /home/trim/wblogs/miss.log 2>/dev/null; do sleep 20; done
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
# 1) зняти ознаки і pk1v пошарово (поправка лишається ВИМКНЕНОЮ, дампимо лише feats)
CORRDUMP=/home/trim/wb2data/corrdump.pkl /home/trim/fs16/bin/whest run \
  --estimator /home/trim/wb2lean/estimator_corrdump.py --dataset $DS --split mini --n-mlps 8 \
  --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 \
  --max-threads 1 --format json > /home/trim/wblogs/corrdump.json 2>/dev/null
echo "dump done" >> /home/trim/wblogs/corr.log
# 2) ЧИСТИЙ послідовний вимір стіни: base і min16 по черзі, без паралелі
E=/home/trim/whest-p2-cumulant-k3/estimators/estimator_v29.py
for cfg in base min16; do
  if [ $cfg = min16 ]; then EXTRA="V26_STRASSEN_MIN=16"; else EXTRA="V26_WARM=1"; fi
  env $EXTRA /home/trim/fs16/bin/whest run --estimator $E --dataset $DS --split mini --n-mlps 4 \
    --flop-budget 2199023255552 --no-residual-wall-time-limit --wall-time-limit 900 \
    --max-threads 1 --format json > /home/trim/wblogs/solo_$cfg.json 2>/dev/null
  echo "solo_$cfg done" >> /home/trim/wblogs/corr.log
done
echo DONE_CORR >> /home/trim/wblogs/corr.log
