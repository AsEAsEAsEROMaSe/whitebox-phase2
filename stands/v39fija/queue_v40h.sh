#!/bin/bash
# 18.10.2026 v40h = лише розпечені рядки (E-D, alpha>3), мертві не викидаються: крок (3) активний.
# Розсуджує: MSE x0.93 у v40/v40d від мертвих рядків чи від кроку (3). Стартує після WEXTRACTDONE. Кінець V40HDONE.
until grep -q "^WEXTRACTDONE" /home/trim/wblogs/v37_runs.log; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null || pgrep -f "[w]extract.py" >/dev/null; do sleep 20; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
for s in s0 s1 s2 s3; do
  DSX=/home/trim/wb2data/dsub/$s ./run2.sh /home/trim/wb2lean/v39fija/estimator_v40_oldrows.py v40h_$s 1 V40_OLD_DEAD=99
done
echo V40HDONE >> /home/trim/wblogs/v37_runs.log
