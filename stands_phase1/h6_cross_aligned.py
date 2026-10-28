import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/h6_cross_aligned.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; K=32     # top-K aligned channels for high-degree CROSS terms
relu=lambda x: torch.clamp(x,min=0.0)
d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
G0=W[0].t()@W[0]; std0=torch.sqrt(torch.diagonal(G0).clamp(min=1e-12))
def ff(w):
    x=w
    for li in range(L): x=relu(x@W[li])
    return x
def He(t,p):
    if p==2: return t*t-1
    if p==3: return t**3-3*t
    if p==4: return t**4-6*t*t+3
    if p==6: return t**6-15*t**4+45*t*t-15
    if p==8: return t**8-28*t**6+210*t**4-420*t*t+105

# rank channels by coupling to outputs (same criterion as before)
log('ranking channels ...')
acc=torch.zeros(D,device=dev)
for c in range(8):
    w=torch.randn(12_500,D,device=dev,generator=torch.Generator(device=dev).manual_seed(500+c))
    Y=0.5*(ff(w)+ff(-w)); s=(w@W[0])/std0; h=He(s,2)
    Yc=Y-Y.mean(0); hc=h-h.mean(0)
    acc+=((hc.t()@Yc)**2).sum(1)/hc.shape[0]**2
top=torch.argsort(acc,descending=True)[:K]
pu=torch.triu_indices(K,K,offset=0)      # i<=j  (for symmetric products)
pf=torch.cartesian_prod(torch.arange(K),torch.arange(K)).t()   # all ordered (for He2xHe4)
NB_DIAG=4*D
NCR = pf.shape[1] + pu.shape[1] + pu.shape[1]   # He2xHe4(ordered) + He4xHe4 + He3xHe3
Ff=NB_DIAG+NCR
log('features: diag He2/4/6/8 {} + cross(He2He4 {} , He4He4 {} , He3He3 {}) = {}'.format(
    NB_DIAG,pf.shape[1],pu.shape[1],pu.shape[1],Ff))
def accum(seeds):
    A=torch.zeros(Ff,Ff,dtype=torch.float64,device=dev); BY=torch.zeros(Ff,D,dtype=torch.float64,device=dev)
    SX=torch.zeros(Ff,dtype=torch.float64,device=dev)
    SY=torch.zeros(D,dtype=torch.float64,device=dev); SY2=torch.zeros(D,dtype=torch.float64,device=dev); N=0
    for c in seeds:
        w=torch.randn(10_000,D,device=dev,generator=torch.Generator(device=dev).manual_seed(c))
        Y=0.5*(ff(w)+ff(-w)); s=(w@W[0])/std0
        h2=He(s,2); h3=He(s,3); h4=He(s,4)
        k2=h2[:,top]; k3=h3[:,top]; k4=h4[:,top]
        cr=torch.cat([k2[:,pf[0]]*k4[:,pf[1]],
                      k4[:,pu[0]]*k4[:,pu[1]],
                      k3[:,pu[0]]*k3[:,pu[1]]],1)
        X=torch.cat([h2,h4,He(s,6),He(s,8),cr],1).double(); Yd=Y.double()
        A+=X.t()@X; BY+=X.t()@Yd; SX+=X.sum(0); SY+=Yd.sum(0); SY2+=(Yd*Yd).sum(0); N+=w.shape[0]
    return [z.cpu() for z in (A,BY,SX,SY,SY2)]+[N]
log('TRAIN 600k ...'); Atr,Btr,SXtr,SYtr,SY2tr,Ntr=accum(range(9000,9060))
log('TEST  400k ...'); Ate,Bte,SXte,SYte,SY2te,Nte=accum(range(20000,20040))
SStot_te=SY2te-SYte*SYte/Nte
alive=SStot_te>1e-9*float(SStot_te.max()); nA=int(alive.sum())
def r2_holdout(idx):
    idx=torch.tensor(idx); m=len(idx)
    Ai=Atr[idx][:,idx]; Bi=Btr[idx]; SXi=SXtr[idx]
    Ac=Ai-torch.outer(SXi,SXi)/Ntr; Bc=Bi-torch.outer(SXi,SYtr)/Ntr
    dsc=torch.sqrt(torch.diagonal(Ac).clamp(min=1e-12))
    As=Ac/torch.outer(dsc,dsc); Bs=Bc/dsc[:,None]
    bs=torch.linalg.solve(As+1e-7*torch.eye(m,dtype=torch.float64),Bs)
    beta=bs/dsc[:,None]
    xbar=SXi/Ntr; ybar=SYtr/Ntr; a=ybar-beta.t()@xbar
    Ae=Ate[idx][:,idx]; Be=Bte[idx]; SXe=SXte[idx]
    quad=((Ae@beta)*beta).sum(0); lin=(beta*Be).sum(0); cross=(beta*SXe[:,None]).sum(0)
    SSres=SY2te-2*a*SYte-2*lin+Nte*a*a+2*a*cross+quad
    return float((1-SSres/SStot_te)[alive].mean())
i_diag=list(range(0,NB_DIAG)); i_cr=list(range(NB_DIAG,Ff))
log('=== HELD-OUT: high-degree CROSS terms on aligned dirs ===')
log('alive {}/{}'.format(nA,D))
rd=r2_holdout(i_diag);        log('  diag He2/4/6/8 aligned      : {:.4f}'.format(rd))
rc=r2_holdout(i_diag+i_cr);   log('  + high-degree CROSS (top-{}) : {:.4f}'.format(K,rc))
log('>>> HONEST incremental from high-degree cross = {:.4f}'.format(rc-rd))
log('done.')
