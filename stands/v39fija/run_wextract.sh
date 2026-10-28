#!/bin/bash
# 18.10.2026 a2: чекає V40DONE (a5) і відсутності whest, тоді одноразово виймає ваги (wextract.py, ~2 ГБ RAM); a5 тримає вікно до WEXTRACTDONE.
until grep -q '^V40DONE' /home/trim/wblogs/v37_runs.log && ! pgrep -f "bin/whest run" >/dev/null; do sleep 20; done
sleep 5
/home/trim/fs123/bin/python3 -I /home/trim/wb2lean/v39fija/wextract.py > /home/trim/wblogs/wextract.log 2>&1
echo "exit $?" >> /home/trim/wblogs/wextract.log
