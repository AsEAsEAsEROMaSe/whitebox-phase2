import json,math,sys
tag=sys.argv[1] if len(sys.argv)>1 else 'fija'
r=[]
for s in range(10):
    try:
        b=json.load(open(f'/home/trim/wblogs/v37_fija_s{s}.json'))['results']; f=json.load(open(f'/home/trim/wblogs/v37_{tag}_s{s}.json'))['results']
    except Exception: continue
    mb,mf=b['per_mlp'][0],f['per_mlp'][0]
    x=(mb['final_layer_mse']/mf['final_layer_mse'],mb['flops_used']/mf['flops_used'],b['adjusted_final_layer_score']/f['adjusted_final_layer_score'],mf['wall_time_s'],mb['wall_time_s'])
    r.append(x); print('s%d %-18s MSE x%.4f flops x%.4f score x%.4f wall %.0f/%.0f'%((s,mf['mlp_name'])+x))
g=lambda i: math.exp(sum(math.log(x[i]) for x in r)/len(r))
print('N=%d geo: MSE x%.4f flops x%.4f score x%.4f  кращих за балом %d'%(len(r),g(0),g(1),g(2),sum(x[2]>1 for x in r)))
