#!/bin/bash
# скан масштабу C·κ₃ (k4_c3) на одній GPU
for s in $2; do ./run_k4t.sh $1 c3s$s EXTRA_K4TERMS="$(cat /home/trim/wb2lean/k4_c3.txt)" EXTRA_K4SCALE=$s; done
