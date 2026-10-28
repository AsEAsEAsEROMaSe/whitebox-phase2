import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG='/mnt/win_d/llm/White-Box/micro_trace.log'
CACHE='/mnt/win_d/llm/White-Box/netbank/micro_trace_cache.pt'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; H=24; SIG=math.sqrt(2.0/D)
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
xh,wh=hermgauss(H)
xn=torch.tensor(xh*SQ2,dtype=torch.float32,device=dev); wn=torch.tensor(wh/math.sqrt(math.pi),dtype=torch.float32,device=dev)
TAPS=[3,7,11,15,19,23,27,30]
def phi(t): return torch.exp(-0.5*t*t)/SQ2PI
def Phi(t): return 0.5*(1+torch.erf(t/SQ2))

def gen(seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    return [torch.randn(D,D,generator=g,device=dev)*SIG for _ in range(L)]

def nngp(W):
    """returns traces at taps + per-layer internal summary features"""
    m=torch.zeros(D,device=dev); K=torch.eye(D,device=dev); tr=[]; feats=[]
    for ell in range(L):
        mu=W[ell].t()@m; C=W[ell].t()@(K-torch.outer(m,m))@W[ell]
        sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-12)); t=mu/sd; Ph=Phi(t); ph=phi(t)
        m=mu*Ph+sd*ph
        rho=(C/torch.outer(sd,sd)).clamp(-0.999999,0.999999); sq=torch.sqrt(1-rho*rho)
        Kg=torch.zeros(D,D,device=dev); sdsq=sd[None,:]*sq
        for a in range(H):
            Ri=relu(mu+sd*xn[a]); gjb=mu[None,:]+sd[None,:]*(rho*xn[a])
            Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*xn[:,None,None])
            Kg+=wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
        Kg[range(D),range(D)]=(mu*mu+sd*sd)*Ph+mu*sd*ph
        K=Kg
        if ell in TAPS:
            off=rho-torch.diag(torch.diagonal(rho))
            tr.append(float(torch.diagonal(K).sum()))
            feats.append([float(t.mean()),float(t.std()),float(sd.mean()),float(sd.std()),
                          float(off.abs().mean()),float((off**2).mean()),
                          float(torch.diagonal(K).std()/torch.diagonal(K).mean())])
    return np.array(tr), np.array(feats)          # (8,), (8,7)

def mc(W,N=400_000,CH=50_000):
    """true traces at taps + measured preactivation kurtosis/skew per tap"""
    s1=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in TAPS]
    s2=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in TAPS]
    s3=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in TAPS]
    s4=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in TAPS]
    ra=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in TAPS]
    n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(90000+c))
        x=torch.cat([z,-z],0); j=0
        for li in range(L):
            h=x@W[li]
            if li in TAPS:
                hd=h.double()
                s1[j]+=hd.sum(0); s2[j]+=(hd*hd).sum(0); s3[j]+=(hd**3).sum(0); s4[j]+=(hd**4).sum(0)
                ra[j]+=(relu(h).double()**2).sum(0); j+=1
            x=relu(h)
        n+=x.shape[0]
    tr=[];ku=[];sk=[]
    for j in range(len(TAPS)):
        e1=s1[j]/n; e2=s2[j]/n; e3=s3[j]/n; e4=s4[j]/n
        var=(e2-e1*e1).clamp(min=1e-12); sd=var.sqrt()
        m3=e3-3*e1*e2+2*e1**3; m4=e4-4*e1*e3+6*e1*e1*e2-3*e1**4
        tr.append(float((ra[j]/n).sum())); ku.append(float((m4/var/var-3).mean())); sk.append(float((m3/sd**3).mean()))
    return np.array(tr),np.array(ku),np.array(sk)

import os
NN=200
if os.path.exists(CACHE):
    d=torch.load(CACHE); TRN,TRT,FE,KU,SK=d['TRN'],d['TRT'],d['FE'],d['KU'],d['SK']; log('cache loaded')
else:
    TRN=[];TRT=[];FE=[];KU=[];SK=[]
    for i in range(NN):
        W=gen(70000+i)
        tn,fe=nngp(W); tt,ku,sk=mc(W)
        TRN.append(tn);TRT.append(tt);FE.append(fe);KU.append(ku);SK.append(sk)
        if i%20==0: log('  net {}/{}'.format(i,NN))
    TRN=np.array(TRN);TRT=np.array(TRT);FE=np.array(FE);KU=np.array(KU);SK=np.array(SK)
    torch.save({'TRN':TRN,'TRT':TRT,'FE':FE,'KU':KU,'SK':SK},CACHE); log('cached')

Y=(TRN-TRT)/TRT                      # (N,8) relative trace error to be predicted
X=np.concatenate([FE.reshape(len(FE),-1),KU,SK],1)     # (N, 8*7+8+8)
log('=== MICRO-MODEL: predict the NNGP trace error from cheap features ===')
log('nets {}  features {}  targets {}'.format(len(Y),X.shape[1],Y.shape[1]))
log('raw RMS rel.err = {:.4e}'.format(np.sqrt((Y**2).mean())))
ntr=int(len(Y)*0.75)
Xtr,Xte=X[:ntr],X[ntr:]; Ytr,Yte=Y[:ntr],Y[ntr:]
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-12
Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
best=None
for lam in [1e-3,1e-2,1e-1,1.0,10.0]:
    A=Xtr.T@Xtr+lam*np.eye(Xtr.shape[1]); B=Xtr.T@(Ytr-Ytr.mean(0))
    W_=np.linalg.solve(A,B)
    pred=Xte@W_+Ytr.mean(0)
    resid=np.sqrt(((Yte-pred)**2).mean()); raw=np.sqrt((Yte**2).mean())
    rho=np.corrcoef(Yte.ravel(),pred.ravel())[0,1]
    log('  ridge {:<6}: held-out RMS {:.4e}  (raw {:.4e})  reduction {:.2f}x   corr {:.4f}'.format(
        lam,resid,raw,raw/resid,rho))
    if best is None or resid<best[1]: best=(lam,resid,raw/resid,rho)
log('')
log('BEST: reduction {:.2f}x  corr {:.4f}'.format(best[2],best[3]))
log('NEEDED: 7.6x  (corr 0.991)')
log('done.')
