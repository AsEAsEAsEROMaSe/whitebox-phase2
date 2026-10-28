#!/bin/bash
# 18.10.2026 white-box-21 (дозвіл юзера «так»): v43 + ECORR_LAM_NEW=1 (λ-таблиця e30_a, MIT) на s0..s9, парно проти v43_s*.
# Черга: дамп (DUMP4DONE) ⟹ a5 профіль і v44 (V44DONE або STOP v44) ⟹ цей прогін ⟹ V43LNDONE ⟹ run_fit90.sh (wextract WX_ALL, fit90).
L=/home/trim/wblogs/v37_runs.log
until grep -qE '^(V44DONE|STOP v44)' $L; do sleep 30; done
while pgrep -f "[w]hest run" >/dev/null; do sleep 20; done
cd /home/trim/wb2lean/v37dr; export WB=/home/trim/fs123
E=/home/trim/wb2lean/v39fija/estimator_v43.py
for s in s0 s1 s2 s3 s4 s5 s6 s7 s8 s9; do DSX=/home/trim/wb2data/dsub/$s ./run2.sh $E v43ln_$s 1 ECORR_LAM_NEW=1; done
echo V43LNDONE >> $L
