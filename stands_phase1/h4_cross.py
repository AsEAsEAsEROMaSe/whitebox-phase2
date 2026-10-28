import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/h4_cross.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; K=48          # K channels whose He2xHe2 cross terms we test
relu=lambda x: torch.clamp(x,min=0.0)
d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
G0=W[0].t()@W[0]; std0=torch.sqrt(torch.diagonal(G0).clamp(min=1e-12))
def ff(w):
    x=w
    for li in range(L): x=relu(x@W[li])
    return x
def base_feats(w):
    s=(w@W[0])/std0
    return s, s*s-1.0, s**4-6*s*s+3.0     # s, He2 (256), He4diag (256)

# ---- pass 1: rank channels by how much their He2 couples to the outputs ----
log('pass1: ranking channels by |Cov(He2_i, Y)| ...')
acc=torch.zeros(D,device=dev)
for c in range(8):
    w=torch.randn(12_500,D,device=dev,generator=torch.Generator(device=dev).manual_seed(500+c))
    Y=0.5*(ff(w)+ff(-w)); s,he2,he4=base_feats(w)
    Yc=Y-Y.mean(0); h=he2-he2.mean(0)
    acc+=((h.t()@Yc)**2).sum(1)/h.shape[0]**2
top=torch.argsort(acc,descending=True)[:K]
log('pass1 done. top-{} channels selected (coupling range {:.2e}..{:.2e})'.format(
    K,float(acc[top[-1]]),float(acc[top[0]])))

# ---- pass 2: accumulate normal equations ----
# features: He2diag(256) + He4diag(256) + He2_i*He2_j cross over top-K (K(K+1)/2)
pairs=torch.triu_indices(K,K,offset=0)
NCROSS=pairs.shape[1]
Ff=D+D+NCROSS
log('feature layout: He2diag {} + He4diag {} + cross {} = {}'.format(D,D,NCROSS,Ff))
NB=600_000; CH=12_500; nch=NB//CH
A=torch.zeros(Ff,Ff,dtype=torch.float64,device=dev); BY=torch.zeros(Ff,D,dtype=torch.float64,device=dev)
SX=torch.zeros(Ff,dtype=torch.float64,device=dev)
SY=torch.zeros(D,dtype=torch.float64,device=dev); SY2=torch.zeros(D,dtype=torch.float64,device=dev); N=0
for c in range(nch):
    w=torch.randn(CH,D,device=dev,generator=torch.Generator(device=dev).manual_seed(9000+c))
    Y=0.5*(ff(w)+ff(-w)); s,he2,he4=base_feats(w)
    hk=he2[:,top]                                   # (n,K)
    cross=hk[:,pairs[0]]*hk[:,pairs[1]]             # (n,NCROSS)  He2_i*He2_j
    X=torch.cat([he2,he4,cross],1).double(); Yd=Y.double()
    A+=X.t()@X; BY+=X.t()@Yd; SX+=X.sum(0); SY+=Yd.sum(0); SY2+=(Yd*Yd).sum(0); N+=w.shape[0]
    if c%12==0: log('  chunk {}/{}'.format(c,nch))
log('accumulated N={}'.format(N))

A=A.cpu(); BY=BY.cpu(); SX=SX.cpu(); SY=SY.cpu(); SY2=SY2.cpu()
SStot=SY2-SY*SY/N
alive=SStot>1e-9*float(SStot.max()); nA=int(alive.sum())
def r2_of(idx):
    idx=torch.tensor(idx)
    Ai=A[idx][:,idx]; Bi=BY[idx]; SXi=SX[idx]
    Ac=Ai-torch.outer(SXi,SXi)/N; Bc=Bi-torch.outer(SXi,SY)/N
    dsc=torch.sqrt(torch.diagonal(Ac).clamp(min=1e-12))
    As=Ac/torch.outer(dsc,dsc); Bs=Bc/dsc[:,None]
    b=torch.linalg.solve(As+1e-8*torch.eye(len(idx),dtype=torch.float64),Bs)
    SSres=SStot-(b*Bs).sum(0)
    return float((1-SSres/SStot)[alive].mean())
i_he2=list(range(0,D)); i_he4=list(range(D,2*D)); i_cr=list(range(2*D,Ff))
r_he2   = r2_of(i_he2)
r_diag  = r2_of(i_he2+i_he4)          # yesterday's model (diag He2+He4)
r_cross = r2_of(i_he2+i_cr)           # He2 + cross only
r_all   = r2_of(i_he2+i_he4+i_cr)     # everything
log('=== degree-4 CROSS terms: the gap in yesterday\'s test ===')
log('alive {}/{}  in-sample R^2 inflation ~{:.4f} (F/N)'.format(nA,D,Ff/N))
log('  He2diag alone                       : {:.4f}'.format(r_he2))
log('  He2diag + He4diag (YESTERDAY)       : {:.4f}'.format(r_diag))
log('  He2diag + He2xHe2 cross (top-{:d})    : {:.4f}'.format(K,r_cross))
log('  ALL (He2 + He4diag + cross)         : {:.4f}'.format(r_all))
log('>>> incremental R^2 from CROSS terms beyond diag = {:.4f}'.format(r_all-r_diag))
log('    cross features used: {}/{} of full degree-4 cross space ({:.1f}%)'.format(
     NCROSS, D*(D+1)//2, 100.0*NCROSS/(D*(D+1)//2)))
log('done.')
