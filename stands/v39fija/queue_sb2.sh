#!/bin/bash
# 19.10.2026 white-box-89: sb2 (ланцюг B + FIJA + tri/nestg/ueye) на q4 після SBDONE. Кінець SB2DONE.
L=/home/trim/wblogs/v37_runs.log
until grep -q "^SBDONE" $L; do sleep 20; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 10 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
D=/home/trim/wb2data/dsub/q4; E=/home/trim/wb2lean/v39fija
DSX=$D ./run2.sh $E/estimator_sb2.py q4_sb2 4
echo SB2DONE >> $L
