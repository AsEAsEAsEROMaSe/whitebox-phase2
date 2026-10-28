import json,glob,os
def load(p):
    d=json.load(open(p))
    def g(o,k):
        if isinstance(o,dict):
            if k in o: return o[k]
            for v in o.values():
                r=g(v,k)
                if r is not None: return r
    per=g(d,"per_mlp"); return {p["mlp_name"]:(p["final_layer_mse"], p["flops_used"]/2199023255552.0) for p in per}
def score(R): return sum(m*max(0.1,u) for m,u in R.values())/len(R)
B=load("/home/trim/wblogs/run_v29_8.json"); sb=score(B)
print("base v29: mse %.4e util %.4f score %.4e" % (sum(m for m,_ in B.values())/8, sum(u for _,u in B.values())/8, sb))
for p in sorted(glob.glob("/home/trim/wblogs/knob_*.json")):
    if os.path.getsize(p)==0: print(os.path.basename(p), "порожній"); continue
    try: R=load(p)
    except Exception as e: print(os.path.basename(p),"ERR",e); continue
    s=score(R); mse=sum(m for m,_ in R.values())/len(R); u=sum(u for _,u in R.values())/len(R)
    wins=sum(R[k][0]*max(0.1,R[k][1]) < B[k][0]*max(0.1,B[k][1]) for k in R)
    print("%-16s mse %.4e (×%.3f) util %.4f (×%.3f) score %.4e (×%.3f) краще %d/%d" % (os.path.basename(p)[5:-5], mse, mse/(sum(m for m,_ in B.values())/8), u, u/(sum(u for _,u in B.values())/8), s, s/sb, wins, len(R)))
