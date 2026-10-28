import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/h6_holdout.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; K=1500
relu=lambda x: torch.clamp(x,min=0.0)
d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
G0=W[0].t()@W[0]; std0=torch.sqrt(torch.diagonal(G0).clamp(min=1e-12))
def ff(w):
    x=w
    for li in range(L): x=relu(x@W[li])
    return x
gu=torch.Generator(device=dev).manual_seed(4242)
U=torch.randn(D,K,device=dev,generator=gu); U=U/U.norm(dim=0,keepdim=True)
def He(t,p):
    if p==6: return t**6-15*t**4+45*t*t-15
    if p==8: return t**8-28*t**6+210*t**4-420*t*t+105
Ff=D+K+K
def accum(seeds):
    A=torch.zeros(Ff,Ff,dtype=torch.float64,device=dev); BY=torch.zeros(Ff,D,dtype=torch.float64,device=dev)
    SX=torch.zeros(Ff,dtype=torch.float64,device=dev)
    SY=torch.zeros(D,dtype=torch.float64,device=dev); SY2=torch.zeros(D,dtype=torch.float64,device=dev); N=0
    for c in seeds:
        w=torch.randn(10_000,D,device=dev,generator=torch.Generator(device=dev).manual_seed(c))
        Y=0.5*(ff(w)+ff(-w))
        s=(w@W[0])/std0
        t=w@U
        X=torch.cat([s*s-1.0,He(t,6),He(t,8)],1).double(); Yd=Y.double()
        A+=X.t()@X; BY+=X.t()@Yd; SX+=X.sum(0); SY+=Yd.sum(0); SY2+=(Yd*Yd).sum(0); N+=w.shape[0]
    return [z.cpu() for z in (A,BY,SX,SY,SY2)]+[N]
log('accumulating TRAIN (600k) ...'); Atr,Btr,SXtr,SYtr,SY2tr,Ntr=accum(range(9000,9060))
log('accumulating TEST  (400k) ...'); Ate,Bte,SXte,SYte,SY2te,Nte=accum(range(20000,20040))
log('train N={}  test N={}'.format(Ntr,Nte))

SStot_te=SY2te-SYte*SYte/Nte
alive=SStot_te>1e-9*float(SStot_te.max()); nA=int(alive.sum())
def r2_holdout(idx):
    idx=torch.tensor(idx); m=len(idx)
    Ai=Atr[idx][:,idx]; Bi=Btr[idx]; SXi=SXtr[idx]
    Ac=Ai-torch.outer(SXi,SXi)/Ntr; Bc=Bi-torch.outer(SXi,SYtr)/Ntr
    dsc=torch.sqrt(torch.diagonal(Ac).clamp(min=1e-12))
    As=Ac/torch.outer(dsc,dsc); Bs=Bc/dsc[:,None]
    bs=torch.linalg.solve(As+1e-8*torch.eye(m,dtype=torch.float64),Bs)
    beta=bs/dsc[:,None]                       # back to raw coords (Ff_sub, D)
    xbar=SXi/Ntr; ybar=SYtr/Ntr
    a=ybar-beta.t()@xbar                      # intercept per output (D,)
    # SSres_test = SY2 -2a.SY -2 b.BY + n a^2 + 2a b.SX + b^T A b   (per output j)
    Ae=Ate[idx][:,idx]; Be=Bte[idx]; SXe=SXte[idx]
    quad=((Ae@beta)*beta).sum(0)              # (D,)
    lin =(beta*Be).sum(0)
    cross=(beta*SXe[:,None]).sum(0)
    SSres=SY2te-2*a*SYte-2*lin+Nte*a*a+2*a*cross+quad
    return float((1-SSres/SStot_te)[alive].mean())
i2=list(range(0,D)); i6=list(range(D,D+K)); i8=list(range(D+K,Ff))
log('=== HELD-OUT R^2 (fit on train, scored on fresh test) ===')
log('alive {}/{}'.format(nA,D))
r2   =r2_holdout(i2);       log('  He2diag alone (whitening baseline) : {:.4f}'.format(r2))
r26  =r2_holdout(i2+i6);    log('  + He6 on {} random dirs           : {:.4f}'.format(K,r26))
r268 =r2_holdout(i2+i6+i8); log('  + He8 too                          : {:.4f}'.format(r268))
log('>>> HONEST incremental R^2 from random high-degree = {:.4f}'.format(r268-r2))
log('done.')
