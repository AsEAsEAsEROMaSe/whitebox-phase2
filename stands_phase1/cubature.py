import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/cubature.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
relu=lambda x: torch.clamp(x,min=0.0)
d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
def fwd_sum(X,wts,CH=8192):
    """weighted sum of a^L over point set X with weights wts (streams in chunks)."""
    acc=torch.zeros(D,dtype=torch.float64,device=dev)
    for i in range(0,X.shape[0],CH):
        x=X[i:i+CH]
        for li in range(L): x=relu(x@W[li])
        acc+=(x.double()*wts[i:i+CH,None]).sum(0)
    return acc

# ---------- ground truth: big whitened-antithetic MC ----------
def gt(N=8_000_000,CH=100_000):
    acc=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(555+c))
        x=torch.cat([z,-z],0)
        cov=(x.t()@x)/x.shape[0]
        ev,V=torch.linalg.eigh(cov.double().cpu()); wh=(V*(1.0/ev.clamp(min=1e-12)).sqrt())@V.t()
        x=x@wh.float().to(dev)
        for li in range(L): x=relu(x@W[li])
        acc+=x.double().sum(0); n+=x.shape[0]
    return acc/n
log('ground truth (8M whitened antithetic) ...')
G=gt(); log('GT done, mean {:.6f}'.format(float(G.mean())))

# ---------- Stroud degree-5 cubature: 2d^2+1 points ----------
log('building Stroud 5-2 rule ...')
A=2.0/(D+2); B=(4.0-D)/(2.0*(D+2)**2); C=1.0/((D+2)**2)
r=math.sqrt(D+2); s=math.sqrt((D+2)/2.0)
pts=[torch.zeros(1,D,device=dev)]; wts=[torch.tensor([A],dtype=torch.float64,device=dev)]
E=torch.eye(D,device=dev)
pts.append(r*E); pts.append(-r*E)
wts.append(torch.full((D,),B,dtype=torch.float64,device=dev))
wts.append(torch.full((D,),B,dtype=torch.float64,device=dev))
iu=torch.triu_indices(D,D,offset=1)
ei=E[iu[0]]; ej=E[iu[1]]
for sa in (1,-1):
    for sb in (1,-1):
        pts.append(s*(sa*ei+sb*ej))
        wts.append(torch.full((iu.shape[1],),C,dtype=torch.float64,device=dev))
X=torch.cat(pts,0); Wt=torch.cat(wts,0)
log('rule: {} points, weight sum = {:.10f} (must be 1), A={:.3e} B={:.3e} C={:.3e}'.format(
    X.shape[0],float(Wt.sum()),A,B,C))
log('all non-center points on sphere radius {:.3f} (typical ||x||={:.3f})'.format(r,math.sqrt(D)))
est_cub=fwd_sum(X,Wt)
mse_cub=float(((est_cub-G)**2).mean())
log('CUBATURE  deg-5 exact, {} pts : MSE = {:.4e}'.format(X.shape[0],mse_cub))

# ---------- v2 at MATCHED budget ----------
NP=X.shape[0]//2
z=torch.randn(NP,D,device=dev,generator=torch.Generator(device=dev).manual_seed(77))
x=torch.cat([z,-z],0)
cov=(x.t()@x)/x.shape[0]
ev,V=torch.linalg.eigh(cov.double().cpu()); wh=(V*(1.0/ev.clamp(min=1e-12)).sqrt())@V.t()
xw=x@wh.float().to(dev)
est_v2=fwd_sum(xw,torch.full((xw.shape[0],),1.0/xw.shape[0],dtype=torch.float64,device=dev))
mse_v2=float(((est_v2-G)**2).mean())
log('v2 (whiten+antithetic), {} pts : MSE = {:.4e}'.format(xw.shape[0],mse_v2))
log('>>> ratio cubature/v2 = {:.3f}   ({})'.format(mse_cub/mse_v2,'CUBATURE WINS' if mse_cub<mse_v2 else 'v2 wins'))
log('note: v2 submitted score uses ~200k pts and reports 3.09e-7')
log('done.')
