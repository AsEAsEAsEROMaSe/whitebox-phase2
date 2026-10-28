import json,numpy as np,sys
D='/home/trim/wb2data/'
def load(f):
    d=json.load(open(D+f)); who='?'
    if 'props' in d:
        s=d['props']['data']['submission']; who=s['participant']['name']+' '+str(s['score'])
        d=s['evaluation'].get('results',s['evaluation'])
    ps=d['public_scores']; pm=d['per_mlp']
    nm={x['mlp_index']:x['mlp_name'] for x in pm}
    fl={x['mlp_index']:x['telemetry']['flops_used'] for x in pm}
    return who,{nm[x['mlp_index']]:(x['scores']['final_layer_mse'],fl[x['mlp_index']]) for x in ps}
R={}
for f in ['grader_331514.json','grader_331976.json','grader_332512.json','grader_332577.json']:
    who,d=load(f); R[f[7:13]]=d; print(f,who,len(d))
names=sorted(set.intersection(*[set(v) for v in R.values()])); print('common',len(names))
M={k:np.array([R[k][n][0] for n in names]) for k in R}
F={k:np.array([R[k][n][1] for n in names]) for k in R}
for k in R: print(k,'meanMSE %.3e  med %.3e  max/min %.1f  flops %.3e'%(M[k].mean(),np.median(M[k]),M[k].max()/M[k].min(),F[k].mean()))
L={k:np.log(M[k]) for k in R}
ks=list(R)
print('corr log MSE між сабмітами'); 
for a in ks: print(a,' '.join('%.3f'%np.corrcoef(L[a],L[b])[0,1] for b in ks))
for a,b in [('331514','332577'),('331514','332512'),('332512','332577')]:
    r=M[a]/M[b]; print(a,'/',b,'ratio mean %.2f med %.2f sd(log) %.3f min %.2f max %.2f'%(r.mean(),np.median(r),np.log(r).std(),r.min(),r.max()))
# share of MSE in top networks
for k in ks:
    s=np.sort(M[k])[::-1]; print(k,'top5 share %.2f top10 %.2f'%(s[:5].sum()/s.sum(),s[:10].sum()/s.sum()))
np.savez('/home/trim/wb2data/permlp_cmp.npz',names=names,**{'M'+k:M[k] for k in ks})
