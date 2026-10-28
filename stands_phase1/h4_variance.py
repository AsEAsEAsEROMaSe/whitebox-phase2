import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/h4_variance.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
relu=lambda x: torch.clamp(x,min=0.0)
d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
# per-channel std of first preactivation z0 = w@W0, w~N(0,I): Cov(z0)=W0^T W0
G0=W[0].t()@W[0]; std0=torch.sqrt(torch.diagonal(G0).clamp(min=1e-12))
log('net0. z0 std range {:.3f}..{:.3f}'.format(float(std0.min()),float(std0.max())))

def features(w):
    # w: (n,D) input ~N(0,I). Even control-variate feats (pop mean 0), directional, all O(1) scale.
    s=(w@W[0])/std0                                    # standardized first-preact channels (n,D)
    he2=s*s-1.0                                         # directional k=2 (256)  [whitening removes these]
    he4=s**4-6*s*s+3.0                                  # directional k=4 (256)  [the new lever]
    return he2,he4

def forward_final(w):
    x=w
    for li in range(L): x=relu(x@W[li])
    return x    # a^L (n,D)

# accumulate normal equations for nested models over antithetic pairs
NB=400_000; CH=12_500; nch=NB//CH
# feature blocks: intercept(1)+rad(2)=3 ; +he2(256) ; +he4(256).  total 515
F=3+D+D
# features WITHOUT intercept (centering handles it): he2(256)+he4(256)=512
Ff=D+D
A=torch.zeros(Ff,Ff,dtype=torch.float64,device=dev)
BY=torch.zeros(Ff,D,dtype=torch.float64,device=dev)
SX=torch.zeros(Ff,dtype=torch.float64,device=dev)
SY=torch.zeros(D,dtype=torch.float64,device=dev); SY2=torch.zeros(D,dtype=torch.float64,device=dev)
N=0
for c in range(nch):
    w=torch.randn(CH,D,device=dev,generator=torch.Generator(device=dev).manual_seed(9000+c))
    ap=forward_final(w); am=forward_final(-w)
    Y=0.5*(ap+am)                     # antithetic even part (n,D)  -> this is what v2 averages
    he2,he4=features(w)               # even feats identical for +/-w
    X=torch.cat([he2,he4],1).double()   # (n,Ff)
    Yd=Y.double()
    A+=X.t()@X; BY+=X.t()@Yd; SX+=X.sum(0)
    SY+=Yd.sum(0); SY2+=(Yd*Yd).sum(0); N+=w.shape[0]
log('accumulated N={} pairs, Ff={}'.format(N,Ff))

# column blocks: rad=[0,2), he2=[2,2+D), he4=[2+D,Ff)  -- SOLVE ON CPU (P104 cusolver->NaN)
A=A.cpu(); BY=BY.cpu(); SX=SX.cpu(); SY=SY.cpu(); SY2=SY2.cpu()
SStot=SY2-SY*SY/N                          # (D,)  total variance*N of Y per neuron
alive=SStot > 1e-9*float(SStot.max())      # exclude dead ReLU neurons (Var=0, v2 nails them)
nA=int(alive.sum())
def r2_of(idx):
    idx=torch.tensor(idx)
    Ai=A[idx][:,idx]; Bi=BY[idx]; SXi=SX[idx]
    Ac=Ai-torch.outer(SXi,SXi)/N           # centered (covariance*N)
    Bc=Bi-torch.outer(SXi,SY)/N
    dsc=torch.sqrt(torch.diagonal(Ac).clamp(min=1e-12))
    As=Ac/torch.outer(dsc,dsc); Bs=Bc/dsc[:,None]
    beta_s=torch.linalg.solve(As+1e-8*torch.eye(len(idx),dtype=torch.float64),Bs)
    SSres=SStot - (beta_s*Bs).sum(0)       # SStot - β_s·(scaled centered XᵀY)
    r2=1-SSres/SStot
    return r2[alive].mean(), SSres         # avg over ALIVE neurons only
r2h2,_=r2_of(list(range(0,D)))             # directional he2 (what whitening takes)
r2h4,_=r2_of(list(range(D,Ff)))            # directional he4 ALONE (new lever)
r2all,_=r2_of(list(range(0,Ff)))           # he2+he4
r2h2=float(r2h2); r2h4=float(r2h4); r2all=float(r2all); incr=r2all-r2h2
varY=float((SStot[alive]/N).mean())
log('=== v2 residual variance decomposition (net0, antithetic even part) ===')
log('alive neurons: {}/{} (dead ReLU excluded — v2 nails them at 0 var)'.format(nA,D))
log('mean Var(Y) per ALIVE neuron (antithetic) = {:.4e}'.format(varY))
log('R^2 of Y explained (avg over ALIVE neurons):')
log('  directional He2 alone (whitening takes): {:.4f}'.format(r2h2))
log('  directional He4 alone (NEW LEVER)      : {:.4f}'.format(r2h4))
log('  He2 + He4 together                     : {:.4f}'.format(r2all))
log('>>> incremental R^2 from He4 BEYOND whitening(He2) = {:.4f}'.format(incr))
log('    => achievable MC mean-variance factor at SAME FLOP ~ x{:.3f}  (v2 MSE 3.09e-7 -> ~{:.2e})'.format(
     1-incr, 3.09e-7*(1-incr)))
log('done.')
