#!/bin/bash
# 19.10.2026 white-box-8c: sbh (ланцюг B sb2 + адаптивна межа розпечених рядків) на q4 після SB3DONE; база q4_sb2. Кінець SBHDONE.
L=/home/trim/wblogs/v37_runs.log
until grep -q SB3DONE $L; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 10 ]; do sleep 30; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
D=/home/trim/wb2data/dsub/q4; E=/home/trim/wb2lean/v39fija
DSX=$D ./run2.sh $E/estimator_sbh.py q4_sbh30 4 SBH_HOT=3.0
DSX=$D ./run2.sh $E/estimator_sbh.py q4_sbh25 4 SBH_HOT=2.5
echo SBHDONE >> $L
