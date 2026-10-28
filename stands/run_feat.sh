#!/bin/bash
cd /home/trim/wb2lean
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-26c6ed1b-62f2-b0d5-05d7-b9e3d1fb3a26
python3 feat.py > /home/trim/wblogs/feat.log 2>&1
