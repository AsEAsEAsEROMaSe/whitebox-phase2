import json,glob,sys,os
rows=[]
for f in sorted(glob.glob('/home/trim/wblogs/early_*.json')):
    try: d=json.load(open(f))
    except Exception as e: print(f,'bad',e); continue
    pm=d['results']['per_mlp']; names=[m['mlp_name'] for m in pm]
    mse=[m['final_layer_mse'] for m in pm]; fl=[m['flops_used'] for m in pm]
    wall=[m.get('wall_time_s') for m in pm]
    rows.append((os.path.basename(f),len(pm),sum(mse)/len(mse),sum(fl)/len(fl),max(wall),mse,fl))
base=[r for r in rows if 'a0_l0' in r[0]]
b=base[0] if base else None
for r in rows:
    mult=max(0.1,r[3]/2.72e11*0+ r[3]/2199023255552)
    s=r[2]*mult
    extra=''
    if b:
        sb=b[2]*max(0.1,b[3]/2199023255552)
        extra=f"  MSE×{r[2]/b[2]:.4f} flops×{r[3]/b[3]:.4f} score×{sb/s:.4f}"
    print(f"{r[0]:22s} n={r[1]} MSE {r[2]:.4e} flops {r[3]:.4e} util {r[3]/2199023255552:.4f} wallmax {r[4]:.0f}{extra}")
if b:
    print("per-net MSE ratio:", [[round(r[5][i]/b[5][i],3) for i in range(len(b[5]))] for r in rows if r is not b])
