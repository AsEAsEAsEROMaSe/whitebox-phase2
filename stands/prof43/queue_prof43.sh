#!/bin/bash
# 18.10.2026 профіль флопів v43 по місцях виклику, 1 мережа s0 (logan). Після DUMP4DONE (дамп white-box-21) і без whest. Кінець PROF43DONE.
until grep -q "^DUMP4DONE" /home/trim/wblogs/v37_runs.log; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
rm -f /home/trim/wblogs/prof43.json.*
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
DSX=/home/trim/wb2data/dsub/s0 ./run2.sh /home/trim/wb2lean/prof43/est43_prof.py prof43_s0 1
echo PROF43DONE >> /home/trim/wblogs/v37_runs.log
