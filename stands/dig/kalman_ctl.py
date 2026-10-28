import numpy as np, torch
dev='cuda'; NS=2**17; CH=2**15
for i in range(3):
    W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],dtype=torch.float32,device=dev)
    S1=torch.zeros(1024,dtype=torch.float64,device=dev); S2=torch.zeros(1024,1024,dtype=torch.float64,device=dev); g=torch.Generator(device=dev).manual_seed(3)
    for _ in range(NS//CH):
        h=torch.randn(CH,1024,generator=g,device=dev)
        for l in range(16): h=torch.relu(h@W[l])
        hd=h.double(); S1+=hd.sum(0); S2+=hd.T@hd
    m=S1/NS; Cv=S2/NS-torch.outer(m,m); lam,U=torch.linalg.eigh(Cv); U=U.flip(1)
    gt=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)['gt_mean'][15].astype(float); pr=np.load('/home/trim/wb2data/pred_%d.npz'%i)['pred'][15].astype(float)
    e=torch.tensor(pr-gt,device=dev); sh=lambda v,k: float(((U.T@v)**2)[:k].sum()/(v*v).sum())
    rng=np.random.default_rng(i); ctl=[sh(e*torch.tensor(rng.choice([-1.,1.],1024),device=dev),16) for _ in range(50)]
    sd=torch.diagonal(Cv).sqrt(); ctl2=[sh(sd*torch.tensor(rng.standard_normal(1024),device=dev),16) for _ in range(50)]
    print('дамп %d: частка похибки в топ-16 напрямках Cov: реальна %.3f · 🔒 ті самі |e| з випадковими знаками %.3f · 🔒 шум ∝ σ_j %.3f · ізотропно 0.016'%(i,sh(e,16),np.mean(ctl),np.mean(ctl2)))
