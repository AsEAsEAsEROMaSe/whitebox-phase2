"""Гіпотеза: дані-залежні флопи Luna/J2W (cv 2.9-3.0%, max/min 1.14, corr між ними 0.94) = виключення насичених нейронів.
Перевірка: розкид частки насичених |t|>th по 100 мережах mini (t з K=2), чи дає cv ціни ~3%.
Передбачення: при th=3..4 cv(1-f) ~1-2%, cv((1-f)^2) ~2-4%. Шанс збігу 50%."""
import numpy as np, torch, math, glob, pyarrow.parquet as pq
dev='cuda'; torch.set_default_dtype(torch.float64)
S='/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a/data/'
def tlayers(W):
    n=W.shape[1]; m=torch.zeros(n,device=dev); C=torch.eye(n,device=dev); T=[]
    for l in range(W.shape[0]):
        Wl=W[l].T; mu=Wl@m; Sg=Wl@C@Wl.T; s=Sg.diag().sqrt(); t=mu/s; T.append(t)
        Ph=0.5*(1+torch.erf(t/math.sqrt(2))); ph=torch.exp(-t*t/2)/math.sqrt(2*math.pi)
        mh=s*(ph+t*Ph); var=s*s*((1+t*t)*Ph+t*ph)-mh*mh
        rho=Sg/(s[:,None]*s[None,:]); rho.fill_diagonal_(0)
        Cn=(s*Ph)[:,None]*(s*Ph)[None,:]*rho + (s*ph)[:,None]*(s*ph)[None,:]*rho**2/2
        Cn.diagonal().copy_(var); C=Cn; m=mh
    return torch.stack(T).cpu().numpy()
out=[]; names=[]
for f in sorted(glob.glob(S+'*.parquet')):
    pf=pq.ParquetFile(f)
    for b in pf.iter_batches(batch_size=1,columns=['mlp_name','weights']):
        d=b.to_pydict(); w=np.asarray(d['weights'][0],dtype=np.float32)
        W=torch.tensor(w.reshape(16,1024,1024),device=dev).double()
        out.append(tlayers(W)); names.append(d['mlp_name'][0]); print(len(out),flush=True,end=' ')
T=np.array(out); np.save('/home/trim/wb2data/satfrac_t100.npy',T.astype(np.float32))
print()
for th in [2.5,3,3.5,4,5]:
    dead=(T<-th).mean((1,2)); lin=(T>th).mean((1,2)); f=dead+lin
    a=1-f
    print('th %.1f  f сер %.3f [%.3f..%.3f]  cv(1-f) %.4f  cv((1-f)^2) %.4f  cv((1-f)^3) %.4f  max/min (1-f)^2 %.3f | dead-only cv((1-d)^2) %.4f'%(th,f.mean(),f.min(),f.max(),a.std()/a.mean(),(a**2).std()/(a**2).mean(),(a**3).std()/(a**3).mean(),(a**2).max()/(a**2).min(),((1-dead)**2).std()/((1-dead)**2).mean()))
