#!/bin/bash
# Драбина глибини Strassen на v29 (дозвіл спонсора 11.09: без ліміту рівнів). 2 офіційні мережі, 1 потік.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DS=/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a
cd /home/trim/wb2lean/bench
for cfg in "V26_STRASSEN=5 V26_STRASSEN_MIN=32" "V26_STRASSEN=6 V26_STRASSEN_MIN=16" "V26_STRASSEN=7 V26_STRASSEN_MIN=8" "V26_STRASSEN=8 V26_STRASSEN_MIN=4"; do
  tag=$(echo $cfg | tr ' =' '__')
  env $cfg /home/trim/fs16/bin/whest run --estimator est_v29.py --dataset $DS --split mini --n-mlps 2 --flop-budget 2199023255552 --format json > v29_$tag.json 2>/dev/null
  python3 - "$cfg" v29_$tag.json <<'PY'
import json,sys
cfg,f=sys.argv[1],sys.argv[2]
try: d=json.load(open(f))
except Exception as e: print(cfg,"NO JSON",e); sys.exit()
r=d["results"]; 
for m in r["per_mlp"]:
    print("%-40s %-18s flops %.4e (%.4f B) wall %6.1f s residual %.3f s exhausted r/t/b %s/%s/%s mse %.4e" % (cfg, m["mlp_name"], m["flops_used"], m["flops_used"]/2**41, m["wall_time_s"], m["residual_wall_time_s"], m["residual_wall_time_exhausted"], m["time_exhausted"], m["budget_exhausted"], m.get("final_layer_mse", float("nan"))))
PY
done
echo DONE_STRASSEN_V29
