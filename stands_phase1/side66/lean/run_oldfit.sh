#!/bin/bash
until grep -q "DONE_OLDFEAT" /home/trim/wblogs/oldfeat.log; do sleep 5; done
cd /home/trim/wb2lean; OMP_NUM_THREADS=8 /home/trim/notqa-venv/bin/python3 -W ignore oldfit.py
