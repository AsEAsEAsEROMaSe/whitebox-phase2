import torch, time, numpy as np, itertools
LOG='/mnt/win_d/llm/White-Box/poly_deep.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
relu=lambda x: torch.clamp(x,min=0.0)
d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
G0=W[0].t()@W[0]; std0=torch.sqrt(torch.diagonal(G0).clamp(min=1e-12))
TAPS=[3,7,11,15,19,23,27,30]          # depths whose collective norm we use

def run(w):
    def fwd(x):
        out=[]
        for li in range(L):
            x=relu(x@W[li])
            if li in TAPS: out.append((x*x).sum(1))
        return x,out
    ap,na=fwd(w); am,nb=fwd(-w)
    Y=0.5*(ap+am)
    Tn=[0.5*(a+b) for a,b in zip(na,nb)]          # antithetic-even deep collective norms
    s=(w@W[0])/std0
    return Y, torch.stack(Tn,1), s                 # Y (n,D), T (n,8), s (n,D)

# feature builder: polynomial modes of the deep collective coords (deg 1..3 + pairwise)
def feats(T):
    z=(T-T.mean(0,keepdim=True))/T.std(0,keepdim=True).clamp(min=1e-9)
    k=z.shape[1]
    cols=[z, z*z, z**3]
    pu=torch.triu_indices(k,k,offset=1)
    cols.append(z[:,pu[0]]*z[:,pu[1]])                       # pairwise deg2
    cols.append(z[:,pu[0]]*z[:,pu[1]]*(z[:,pu[0]]+z[:,pu[1]]))  # pairwise deg3
    return torch.cat(cols,1)

# reference: input Hermite family (aligned, deg 2..8) as measured before
def feats_input(s):
    return torch.cat([s*s-1, s**4-6*s*s+3, s**6-15*s**4+45*s*s-15, s**8-28*s**6+210*s**4-420*s*s+105],1)

log('probing feature dims ...')
w=torch.randn(2000,D,device=dev,generator=torch.Generator(device=dev).manual_seed(1))
Y_,T_,s_=run(w); FD=feats(T_).shape[1]; FI=feats_input(s_).shape[1]
log('deep-poly feats = {} | input-Hermite feats = {}'.format(FD,FI))
Ff=FD+FI
def accum(seeds,CH=10000):
    A=torch.zeros(Ff,Ff,dtype=torch.float64,device=dev); BY=torch.zeros(Ff,D,dtype=torch.float64,device=dev)
    SX=torch.zeros(Ff,dtype=torch.float64,device=dev)
    SY=torch.zeros(D,dtype=torch.float64,device=dev); SY2=torch.zeros(D,dtype=torch.float64,device=dev); N=0
    for c in seeds:
        w=torch.randn(CH,D,device=dev,generator=torch.Generator(device=dev).manual_seed(c))
        Y,T,s=run(w)
        X=torch.cat([feats(T),feats_input(s)],1).double(); Yd=Y.double()
        A+=X.t()@X; BY+=X.t()@Yd; SX+=X.sum(0); SY+=Yd.sum(0); SY2+=(Yd*Yd).sum(0); N+=CH
    return [z.cpu() for z in (A,BY,SX,SY,SY2)]+[N]
log('TRAIN 400k ...'); Atr,Btr,SXtr,SYtr,SY2tr,Ntr=accum(range(3000,3040))
log('TEST  300k ...'); Ate,Bte,SXte,SYte,SY2te,Nte=accum(range(6000,6030))
SStot=SY2te-SYte*SYte/Nte
alive=SStot>1e-9*float(SStot.max()); nA=int(alive.sum())
def r2(idx):
    idx=torch.tensor(idx); m=len(idx)
    Ai=Atr[idx][:,idx]; Bi=Btr[idx]; SXi=SXtr[idx]
    Ac=Ai-torch.outer(SXi,SXi)/Ntr; Bc=Bi-torch.outer(SXi,SYtr)/Ntr
    dsc=torch.sqrt(torch.diagonal(Ac).clamp(min=1e-12))
    bs=torch.linalg.solve(Ac/torch.outer(dsc,dsc)+1e-8*torch.eye(m,dtype=torch.float64),Bc/dsc[:,None])
    beta=bs/dsc[:,None]
    xbar=SXi/Ntr; a=SYtr/Ntr-beta.t()@xbar
    Ae=Ate[idx][:,idx]; Be=Bte[idx]; SXe=SXte[idx]
    SSres=SY2te-2*a*SYte-2*(beta*Be).sum(0)+Nte*a*a+2*a*(beta*SXe[:,None]).sum(0)+((Ae@beta)*beta).sum(0)
    return float((1-SSres/SStot)[alive].mean())
i_d=list(range(0,FD)); i_i=list(range(FD,Ff))
log('=== POLYNOMIAL MODES: deep collective coords vs input Hermite (held-out) ===')
log('alive {}/{}'.format(nA,D))
pass
log('  input Hermite measured              : {:.4f}'.format(r2(i_i)))
log('  DEEP poly modes ({} feats)          : {:.4f}'.format(FD,r2(i_d)))
log('  both together                       : {:.4f}'.format(r2(i_d+i_i)))
log('reference: single deep scalar ||a31||^2 stratification ceiling was 0.297')
log('done.')
