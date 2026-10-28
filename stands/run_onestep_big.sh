#!/bin/bash
# пломба масштабування: onestep з N=2^23 (×8 семплів) на his_useall; якщо похибка народження ~3% T стоїть, вона реальна; якщо падає ~√8, це шум розкладу
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1
cd /home/trim/wb2lean; L=/home/trim/wblogs/onestep_big_d$2.log
echo "== N=2^23 useall dump $2 $(date +%H:%M)" > $L
HIS=/home/trim/wb2lean/his_useall_dump%d.npz N=8388608 /home/trim/notqa-venv/bin/python3 -W ignore onestep.py $2 2>&1 | grep -vE "Warning|warn" | grep -E "^l=|pass" >> $L
echo DONE_BIG_$2 >> $L
