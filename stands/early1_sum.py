import json,glob,os
b=json.load(open('/home/trim/wblogs/early_a0_l0.json'))['results']['per_mlp'][0]
print(f"base          MSE {b['final_layer_mse']:.4e} flops {b['flops_used']:.4e}")
for f in sorted(glob.glob('/home/trim/wblogs/early1_*.json')):
    try: m=json.load(open(f))['results']['per_mlp'][0]
    except Exception as e: print(f,'bad'); continue
    r=m['final_layer_mse']/b['final_layer_mse']; c=m['flops_used']/b['flops_used']
    print(f"{os.path.basename(f):22s} MSE×{r:.3f} flops×{c:.4f} score×{1/(r*c):.3f}")
