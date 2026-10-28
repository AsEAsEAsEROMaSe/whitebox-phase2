import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/h6_random.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; K=1500       # K random directions for high-degree Hermite features
relu=lambda x: torch.clamp(x,min=0.0)
d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
G0=W[0].t()@W[0]; std0=torch.sqrt(torch.diagonal(G0).clamp(min=1e-12))
def ff(w):
    x=w
    for li in range(L): x=relu(x@W[li])
    return x
# random unit directions: u.w ~ N(0,1) exactly for w~N(0,I)
gu=torch.Generator(device=dev).manual_seed(4242)
U=torch.randn(D,K,device=dev,generator=gu); U=U/U.norm(dim=0,keepdim=True)

def He(t,p):
    if p==4: return t**4-6*t*t+3
    if p==6: return t**6-15*t**4+45*t*t-15
    if p==8: return t**8-28*t**6+210*t**4-420*t*t+105
    raise ValueError

# features: He2 diag in first-layer basis (256, what whitening takes) + He6(rand) + He8(rand)
Ff=D+K+K
log('features: He2diag {} + He6rand {} + He8rand {} = {}'.format(D,K,K,Ff))
NB=600_000; CH=10_000; nch=NB//CH
A=torch.zeros(Ff,Ff,dtype=torch.float64,device=dev); BY=torch.zeros(Ff,D,dtype=torch.float64,device=dev)
SX=torch.zeros(Ff,dtype=torch.float64,device=dev)
SY=torch.zeros(D,dtype=torch.float64,device=dev); SY2=torch.zeros(D,dtype=torch.float64,device=dev); N=0
for c in range(nch):
    w=torch.randn(CH,D,device=dev,generator=torch.Generator(device=dev).manual_seed(9000+c))
    Y=0.5*(ff(w)+ff(-w))
    s=(w@W[0])/std0; he2=s*s-1.0
    t=w@U                                  # (n,K) each ~N(0,1)
    X=torch.cat([he2,He(t,6),He(t,8)],1).double(); Yd=Y.double()
    A+=X.t()@X; BY+=X.t()@Yd; SX+=X.sum(0); SY+=Yd.sum(0); SY2+=(Yd*Yd).sum(0); N+=w.shape[0]
    if c%15==0: log('  chunk {}/{}'.format(c,nch))
log('accumulated N={}'.format(N))
A=A.cpu(); BY=BY.cpu(); SX=SX.cpu(); SY=SY.cpu(); SY2=SY2.cpu()
SStot=SY2-SY*SY/N; alive=SStot>1e-9*float(SStot.max()); nA=int(alive.sum())
def r2_of(idx):
    idx=torch.tensor(idx)
    Ai=A[idx][:,idx]; Bi=BY[idx]; SXi=SX[idx]
    Ac=Ai-torch.outer(SXi,SXi)/N; Bc=Bi-torch.outer(SXi,SY)/N
    dsc=torch.sqrt(torch.diagonal(Ac).clamp(min=1e-12))
    As=Ac/torch.outer(dsc,dsc); Bs=Bc/dsc[:,None]
    b=torch.linalg.solve(As+1e-8*torch.eye(len(idx),dtype=torch.float64),Bs)
    return float((1-(SStot-(b*Bs).sum(0))/SStot)[alive].mean())
i2=list(range(0,D)); i6=list(range(D,D+K)); i8=list(range(D+K,Ff))
r_2   = r2_of(i2)
r_26  = r2_of(i2+i6)
r_268 = r2_of(i2+i6+i8)
log('=== BOT IDEA TEST: random high-degree Hermite feature control variate ===')
log('alive {}/{}   in-sample R^2 inflation ~{:.4f} (F/N)'.format(nA,D,Ff/N))
log('  He2diag alone (whitening baseline)     : {:.4f}'.format(r_2))
log('  + He6 on {} random directions          : {:.4f}'.format(K,r_26))
log('  + He8 too                              : {:.4f}'.format(r_268))
log('>>> incremental R^2 from random high-degree feats = {:.4f}'.format(r_268-r_2))
log('    (random dirs sample {}/{:.1e} of degree-6 mode space)'.format(K,256.0**6))
log('done.')
