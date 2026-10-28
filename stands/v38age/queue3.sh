#!/bin/bash
# 18.10.2026: перевірка вихідників пакетів БЕЗ змінних оточення (за замовчуванням) на s0; мусять збігтися з v9q_s0 і age_l12a6_s0
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh /home/trim/wb2sub/v37fix/estimator_v37_fix.py pkgchk37_s0 1
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh /home/trim/wb2sub/v38age/estimator_v38_age.py pkgchk38_s0 1
echo PKGCHKDONE >> /home/trim/wblogs/v37_runs.log
