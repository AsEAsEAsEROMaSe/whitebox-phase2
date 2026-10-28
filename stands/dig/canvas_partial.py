import numpy as np, torch, sys
sys.argv=[sys.argv[0],'0','6']; exec(open('/home/trim/wb2lean/dig/k2ladder.py').read().split("i=int(sys.argv[1])")[0])
out={L:[] for L in [8,12,15]}
for i in range(8):
    W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],device=dev).double()
    gt=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)['gt_mean'].astype(float)
    m2=run(W,'full').cpu().numpy(); m3=np.load('/home/trim/wb2data/pred_%d.npz'%i)['pred'].astype(float)
    for L in out:
        keep=np.abs(m3[L])>1e-3*np.abs(m3[L]).max()
        lm=np.log(np.abs(m3[L][keep])); ld=np.log(np.abs(m3[L]-m2[L])[keep]+1e-18); le=np.log(np.abs(m3[L]-gt[L])[keep]+1e-18)
        X=np.stack([np.ones_like(lm),lm],1)
        rd=ld-X@np.linalg.lstsq(X,ld,rcond=None)[0]; re=le-X@np.linalg.lstsq(X,le,rcond=None)[0]
        e2=(m3[L]-gt[L])[keep]**2; k=len(e2)//10
        top=e2[np.argsort(-rd)[:k]].sum()/e2.sum()
        out[L].append((np.corrcoef(rd,re)[0,1], top, keep.mean()))
print('шар | часткова corr(log|d3|, log|e3|) за вирахуванням масштабу | MSE у топ-10% за залишком |d3| (випадкові 0.10) | частка нейронів')
for L,v in out.items(): a=np.array(v).mean(0); print('%2d  |  %+.3f  |  %.3f  |  %.2f'%(L,*a))
