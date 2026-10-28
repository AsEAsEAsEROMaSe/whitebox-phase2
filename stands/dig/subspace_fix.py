"""Похибка v34 узгоджена вздовж топ-k напрямків Cov(h15). Чи передбачувана її проєкція з того, що ланцюг знає?
Кандидати (проєктовані на топ-k): d3 = m3−m2, d2 = m2−m1, m3 (масштаб). LOO по 8 дампах: e_k ≈ a·P d3 + b·P d2 + c·P m3.
Оракул: прибрати P_k e повністю. Передбачення до запуску: LOO бере ≤10% оракула (корпус: ознаки ланцюга не бачать його похибки);
шанс > 30% оракула: 10%."""
import numpy as np, torch, sys
sys.argv=[sys.argv[0],'0','6']; exec(open('/home/trim/wb2lean/dig/k2ladder.py').read().split("i=int(sys.argv[1])")[0])
NS=2**17; CH=2**15; D=[]
for i in range(8):
    W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],device=dev).double(); Wf=W.float()
    S1=torch.zeros(1024,device=dev); S2=torch.zeros(1024,1024,device=dev); g=torch.Generator(device=dev).manual_seed(3)
    for _ in range(NS//CH):
        h=torch.randn(CH,1024,generator=g,device=dev,dtype=torch.float32)
        for l in range(16): h=torch.relu(h@Wf[l])
        hd=h.double(); S1+=hd.sum(0); S2+=hd.T@hd
    m=S1/NS; Cv=S2/NS-torch.outer(m,m); lam,U=torch.linalg.eigh(Cv); U=U.flip(1).cpu().numpy()
    gt=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)['gt_mean'][15].astype(float); m3=np.load('/home/trim/wb2data/pred_%d.npz'%i)['pred'][15].astype(float)
    m1=run(W,'diag')[15].cpu().numpy(); m2=run(W,'full')[15].cpu().numpy()
    D.append(dict(U=U,e=m3-gt,d3=m3-m2,d2=m2-m1,m3=m3,gt=gt))
for k in [16,64,128]:
    base=np.mean([np.mean(d['e']**2) for d in D]); orc=[];cos=[];lo=[]
    for d in D:
        P=d['U'][:,:k]; pe=P@(P.T@d['e']); orc.append(np.mean((d['e']-pe)**2))
        cos.append([np.dot(P.T@d['e'],P.T@d[f])/np.linalg.norm(P.T@d['e'])/np.linalg.norm(P.T@d[f]) for f in ['d3','d2','m3']])
    for t in range(8):
        X=[];Y=[]
        for j in range(8):
            if j==t: continue
            P=D[j]['U'][:,:k]; X.append(np.stack([P@(P.T@D[j][f]) for f in ['d3','d2','m3']],1)); Y.append(P@(P.T@D[j]['e']))
        c=np.linalg.lstsq(np.concatenate(X),np.concatenate(Y),rcond=None)[0]
        P=D[t]['U'][:,:k]; corr=np.stack([P@(P.T@D[t][f]) for f in ['d3','d2','m3']],1)@c; lo.append(np.mean((D[t]['e']-corr)**2))
    cs=np.array(cos).mean(0)
    print('топ-%3d: база %.3e · оракул %.3e (x%.2f) · LOO фіт %.3e (x%.3f) · cos(P e, P d3/d2/m3) %+.2f %+.2f %+.2f'%(k,base,np.mean(orc),base/np.mean(orc),np.mean(lo),base/np.mean(lo),*cs))
