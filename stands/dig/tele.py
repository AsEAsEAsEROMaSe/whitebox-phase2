import json,numpy as np
D='/home/trim/wb2data/'
def load(f):
    d=json.load(open(D+f))
    if 'props' in d: d=d['props']['data']['submission']['evaluation'].get('results',d['props']['data']['submission']['evaluation'])
    return d['per_mlp']
keys=['flops_used','participant_predict_wall_s','flopscope_server_kernel_wall_s','flopscope_server_overhead_wall_s','residual_wall_time_s','flopscope_server_comms_wall_s','participant_setup_wall_s']
for f,nm in [('grader_331976.json','ми v33?'),('grader_331514.json','ми'),('grader_332512.json','J2W'),('grader_332577.json','Luna')]:
    pm=load(f); T={k:np.array([x['telemetry'][k] for x in pm]) for k in keys}
    F=T['flops_used']; K=T['flopscope_server_kernel_wall_s']; O=T['flopscope_server_overhead_wall_s']
    print('%-8s n=%d flops %.3e (min %.3e max %.3e cv %.4f) kernel %.1fs overhead %.2fs comms %.2fs resid %.3fs  GF/s kernel %.0f  O/K %.3f'%(nm,len(pm),F.mean(),F.min(),F.max(),F.std()/F.mean(),K.mean(),O.mean(),T['flopscope_server_comms_wall_s'].mean(),T['residual_wall_time_s'].mean(),F.mean()/K.mean()/1e9,O.mean()/K.mean()))
    print('   distinct flops values:',len(set(F.tolist())), ' corr(kernel,flops) %.2f'%np.corrcoef(K,F)[0,1] if F.std()>0 else '')
print('--- помережно (публічні 50)')
def full(f):
    d=json.load(open(D+f))
    if 'props' in d: d=d['props']['data']['submission']['evaluation'].get('results',d['props']['data']['submission']['evaluation'])
    nm={x['mlp_index']:x['mlp_name'] for x in d['per_mlp']}
    fl={x['mlp_name']:x['telemetry']['flops_used'] for x in d['per_mlp']}
    ms={nm[x['mlp_index']]:x['scores']['final_layer_mse'] for x in d['public_scores']}
    return fl,ms
A={k:full(f) for k,f in [('us','grader_331514.json'),('J2W','grader_332512.json'),('Luna','grader_332577.json')]}
names=sorted(A['us'][1]); 
F={k:np.array([A[k][0][n] for n in names]) for k in A}; M={k:np.log([A[k][1][n] for n in names]) for k in A}
c=lambda a,b: np.corrcoef(a,b)[0,1]
print('corr flops Luna~J2W %.2f'%c(F['Luna'],F['J2W']))
for k in ['Luna','J2W']:
    print(k,'corr(flops, own logMSE) %.2f  corr(flops, our logMSE) %.2f  corr(flops, flops us) %.2f'%(c(F[k],M[k]),c(F[k],M['us']),c(F[k],F['us'])))
# all 100 nets flops correlation (incl private)
fl={k:A[k][0] for k in A}; allnm=sorted(fl['us'])
print('all100 corr flops Luna~J2W %.2f'%c(np.array([fl['Luna'][n] for n in allnm]),np.array([fl['J2W'][n] for n in allnm])))
