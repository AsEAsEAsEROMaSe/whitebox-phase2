#!/bin/bash
# послідовно на одній GPU: варіанти EXTRA_K4TERMS зі списків-файлів
for t in k4_c3 k4_21 k4_213 k4_all; do ./run_k4t.sh $1 $t EXTRA_K4TERMS="$(cat /home/trim/wb2lean/$t.txt)"; done; echo DONE_K4B >> /home/trim/wblogs/k4t.log
