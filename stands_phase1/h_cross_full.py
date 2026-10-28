import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/h_cross_full.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; K=128        # top-K channels -> K(K+1)/2 = 8256 cross features (25% of space)
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
    if p==4: return t**4-6*t*t+3
    if p==6: return t**6-15*t**4+45*t*t-15
    if p==8: return t**8-28*t**6+210*t**4-420*t*t+105

log('ranking channels by coupling ...')
acc=torch.zeros(D,device=dev)
for c in range(8):
    w=torch.randn(12_500,D,device=dev,generator=torch.Generator(device=dev).manual_seed(500+c))
    Y=0.5*(ff(w)+ff(-w)); s=(w@W[0])/std0; h=He(s,2)
    Yc=Y-Y.mean(0); hc=h-h.mean(0)
    acc+=((hc.t()@Yc)**2).sum(1)/hc.shape[0]**2
top=torch.argsort(acc,descending=True)[:K]
pu=torch.triu_indices(K,K,offset=0)      # i<=j
NDIAG=4*D; NCR=pu.shape[1]; Ff=NDIAG+NCR
log('features: diag He2/4/6/8 {} + He2xHe2 cross top-{} = {}  (cross covers {:.1f}% of full {} space)'.format(
    NDIAG,K,Ff,100.0*NCR/(D*(D+1)//2),D*(D+1)//2))

def accum(seeds,CH=5000):
    A=torch.zeros(Ff,Ff,dtype=torch.float64,device=dev); BY=torch.zeros(Ff,D,dtype=torch.float64,device=dev)
    SX=torch.zeros(Ff,dtype=torch.float64,device=dev)
    SY=torch.zeros(D,dtype=torch.float64,device=dev); SY2=torch.zeros(D,dtype=torch.float64,device=dev); N=0
    for i,c in enumerate(seeds):
        w=torch.randn(CH,D,device=dev,generator=torch.Generator(device=dev).manual_seed(c))
        Y=0.5*(ff(w)+ff(-w)); s=(w@W[0])/std0
        h2=He(s,2); k2=h2[:,top]
        X=torch.cat([h2,He(s,4),He(s,6),He(s,8),k2[:,pu[0]]*k2[:,pu[1]]],1)   # fp32 (P104: fp64 = 1/32 speed)
        Yd=Y.double()
        A+=(X.t()@X).double(); BY+=(X.t()@Y).double(); SX+=X.sum(0).double()
        SY+=Yd.sum(0); SY2+=(Yd*Yd).sum(0); N+=CH
        if i%20==0: log('   chunk {}/{}'.format(i,len(seeds)))
    return [z.cpu() for z in (A,BY,SX,SY,SY2)]+[N]
log('TRAIN 600k ...'); Atr,Btr,SXtr,SYtr,SY2tr,Ntr=accum(range(9000,9120))
log('TEST  400k ...'); Ate,Bte,SXte,SYte,SY2te,Nte=accum(range(20000,20080))
log('train N={} test N={}'.format(Ntr,Nte))
SStot_te=SY2te-SYte*SYte/Nte
alive=SStot_te>1e-9*float(SStot_te.max()); nA=int(alive.sum())
def r2_holdout(idx,ridge=1e-6):
    idx=torch.tensor(idx); m=len(idx)
    Ai=Atr[idx][:,idx]; Bi=Btr[idx]; SXi=SXtr[idx]
    Ac=Ai-torch.outer(SXi,SXi)/Ntr; Bc=Bi-torch.outer(SXi,SYtr)/Ntr
    dsc=torch.sqrt(torch.diagonal(Ac).clamp(min=1e-12))
    As=Ac/torch.outer(dsc,dsc); Bs=Bc/dsc[:,None]
    bs=torch.linalg.solve(As+ridge*torch.eye(m,dtype=torch.float64),Bs)
    beta=bs/dsc[:,None]
    xbar=SXi/Ntr; ybar=SYtr/Ntr; a=ybar-beta.t()@xbar
    Ae=Ate[idx][:,idx]; Be=Bte[idx]; SXe=SXte[idx]
    quad=((Ae@beta)*beta).sum(0); lin=(beta*Be).sum(0); cross=(beta*SXe[:,None]).sum(0)
    SSres=SY2te-2*a*SYte-2*lin+Nte*a*a+2*a*cross+quad
    return float((1-SSres/SStot_te)[alive].mean())
i_d=list(range(0,NDIAG)); i_c=list(range(NDIAG,Ff))
log('=== BOT-PROPOSED FALSIFICATION: full-ish He2xHe2 cross space ===')
log('alive {}/{}   F/N = {:.4f}'.format(nA,D,Ff/Ntr))
rd=r2_holdout(i_d);      log('  diag He2/4/6/8 (reference, was 0.1722) : {:.4f}'.format(rd))
rc=r2_holdout(i_d+i_c);  log('  + He2xHe2 cross, top-{} ({} feats)   : {:.4f}'.format(K,NCR,rc))
log('>>> HONEST incremental from big cross space = {:.4f}'.format(rc-rd))
log('    bot criterion: if total stays <=1.05% beyond whitening -> saturation confirmed')
log('done.')
