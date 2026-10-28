"""Автороутинг (юзер 15.10): чи «море» κ4 нейрона j (κ4 − діаг − пари) сидить у його топ-k входах за |w_aj|?
q_k = Σ_{a∈top_k(j)} w_aj h_a; море_k = κ4(q_k) − [діаг + 3 пари κ22 + 4 пари κ31] на тих самих входах. corr(море_k, море_повне) по нейронах.
Контроль: k випадкових входів. Передбачення до запуску: k=128 corr ~0.6, k=32 ~0.3; шанс k=32 > 0.7: 15%."""
import sys, numpy as np, torch
dev='cuda'; i=0; L=12
W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],dtype=torch.float32,device=dev)
N=2**20; CH=2**15; KS=[32,128,512]
Wd=W[L].double(); order=torch.argsort(-Wd.abs(),dim=0); g=torch.Generator(device=dev).manual_seed(5)
masks={}
for k in KS:
    M=torch.zeros_like(Wd); M.scatter_(0,order[:k],1.0); masks['top%d'%k]=M
    R=torch.zeros_like(Wd); R.scatter_(0,torch.argsort(torch.rand(Wd.shape,generator=g,device=dev,dtype=torch.float64),dim=0)[:k],1.0); masks['rnd%d'%k]=R
masks['full']=torch.ones_like(Wd)
def gen(seed):
    gg=torch.Generator(device=dev).manual_seed(seed)
    for _ in range(N//CH):
        h=torch.randn(CH,1024,generator=gg,device=dev)
        for l in range(L): h=torch.relu(h@W[l])
        yield h
S1=torch.zeros(1024,dtype=torch.float64,device=dev)
for h in gen(1): S1+=h.double().sum(0)
mu=(S1/N).float()
S2=torch.zeros(1024,1024,dtype=torch.float64,device=dev); S22=torch.zeros_like(S2); S31=torch.zeros_like(S2); S2d=torch.zeros_like(S1)
P={nm:[torch.zeros_like(S1) for _ in range(4)] for nm in masks}
Wm={nm:(Wd*M).float() for nm,M in masks.items()}
for h in gen(1):
    x=h-mu; x2=x*x; S2+=(x.T@x).double(); S22+=(x2.T@x2).double(); S31+=((x2*x).T@x).double(); S2d+=x2.double().sum(0)
    for nm,Wk in Wm.items():
        q=(x@Wk).double()
        for k in range(4): P[nm][k]+=(q**(k+1)).sum(0)
C=S2/N; v=S2d/N; K22=S22/N-torch.outer(v,v)-2*C*C; K31=S31/N-3*v[:,None]*C
d=torch.diagonal(K22).clone(); K22o=K22.clone(); K22o.fill_diagonal_(0); K31o=K31.clone(); K31o.fill_diagonal_(0)
sea={}
for nm,M in masks.items():
    w=Wd*M; w2=w*w; m1,m2,m3,m4=[x/N for x in P[nm]]
    k4=m4-4*m1*m3+6*m1**2*m2-3*m1**4-3*(m2-m1**2)**2
    pair=(w2*w2*d[:,None]).sum(0)+3*((K22o@w2)*w2).sum(0)+4*(((w**3).T@K31o)*w.T).sum(1)
    sea[nm]=k4-pair
cc=lambda a,b: float(torch.corrcoef(torch.stack([a,b]))[0,1])
print('шар %d: sd моря по нейронах (повне) %.3e'%(L,float(sea['full'].std())))
for k in KS:
    print('k=%4d: corr(море_топk, море) %.3f  частка sd %.3f · 🔒випадкові k: corr %.3f частка sd %.3f'%(k,cc(sea['top%d'%k],sea['full']),float(sea['top%d'%k].std()/sea['full'].std()),cc(sea['rnd%d'%k],sea['full']),float(sea['rnd%d'%k].std()/sea['full'].std())))
