#!/bin/bash
# пропуск народжень κ₃ на шарах 0..K-1 (reg, без конфайнменту), 8 дампів
for sk in "$2"; do :; done
for sk in $2; do ./run_k4t.sh $1 skip_$sk SKIP_BIRTH=$sk; done
