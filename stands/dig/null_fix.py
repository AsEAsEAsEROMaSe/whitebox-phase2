"""Ідея юзера (крони -> корені): напрямки виходу з ~нульовою дисперсією = null(W15[живі,:]) (мертві нейрони шару 14).
У них u^T out(z) майже стале ⟹ N семплів дають u^T E[out] майже точно. Виправлення: pred + P_null (mc - pred).
Живі = t14 >= -TH (t з банку; у сабміті було б з ланцюга). Передбачення: x1.03-1.08 при N=16..256, ціна <0.5% B.
Контроль: проєктор на ВИПАДКОВИЙ підпростір тієї ж розмірності мусить шкодити (МК там шумна)."""
import numpy as np, torch
dev='cuda'; torch.set_default_dtype(torch.float64)
res={}
for i in range(8):
    W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],device=dev).double()
    kp=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)
    mu=kp['pre_mean'][14]; t14=mu/np.sqrt(kp['pre_m2'][14]-mu**2)
    z=np.load('/home/trim/wb2data/pred_%d.npz'%i); pred=torch.tensor(z['pred'][15].astype(float),device=dev); gt=torch.tensor(z['gt'][15].astype(float),device=dev)
    base=float(((pred-gt)**2).mean())
    for seed in range(3):
        g=torch.Generator(device=dev).manual_seed(1000*i+seed)
        Zs=torch.randn(256,1024,device=dev,generator=g); a=Zs
        for l in range(16):
            p=a@W[l]; a=torch.relu(p) if l<15 else p
        for TH in [2.0,3.0,4.0]:
            alive=torch.tensor(t14>=-TH,device=dev)
            A=W[15][alive,:]                      # (n_alive, 1024)
            Q,_=torch.linalg.qr(A.T)              # range(A^T) = ортодоповнення null(A)
            def Pn(v): return v-Q@(Q.T@v)
            Rr=torch.linalg.qr(torch.randn(1024,1024-int(alive.sum()),device=dev,generator=g))[0]
            for N in [16,64,256]:
                mc=a[:N].mean(0)
                new=pred+Pn(mc-pred); ctl=pred+Rr@(Rr.T@(mc-pred))
                res.setdefault((TH,N),[]).append((base,float(((new-gt)**2).mean()),float(((ctl-gt)**2).mean()),1024-int(alive.sum())))
for (TH,N),v in sorted(res.items()):
    v=np.array(v); print('TH %.1f N=%3d (%.2f%%B): null-напрямків %d | MSE x%.4f [мін %.3f макс %.3f] | контроль випадковий підпростір x%.4f'%(TH,N,100*N*3.36e7/2.199e12,v[:,3].mean(),v[:,0].sum()/v[:,1].sum(),(v[:,0]/v[:,1]).min(),(v[:,0]/v[:,1]).max(),v[:,0].sum()/v[:,2].sum()))
