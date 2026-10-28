#!/bin/bash
# 18.10.2026 white-box-21: ra32 зупинено після s2 (16G RSS, OOMGUARD на s1, правило юзера про RAM). Дочекатися кінця v45ra_s2 і написати RA32DONE.
L=/home/trim/wblogs/v37_runs.log
until grep -qE '^(DONE|OOMGUARD) v45ra_s2' $L && ! pgrep -f "[w]hest run.*estimator_v45.py" >/dev/null; do sleep 15; done
echo "RA32 STOPPED after s2 (RAM 16G)" >> $L
echo RA32DONE >> $L
