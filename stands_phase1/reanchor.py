import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG='/mnt/win_d/llm/White-Box/reanchor.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; H=24
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
xh,wh=hermgauss(H)
xn=torch.tensor(xh*SQ2,dtype=torch.float32,device=dev); wn=torch.tensor(wh/math.sqrt(math.pi),dtype=torch.float32,device=dev)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
TAPS=[3,7,11,15,19,23,27,30]
def phi(t): return torch.exp(-0.5*t*t)/SQ2PI
def Phi(t): return 0.5*(1+torch.erf(t/SQ2))

def true_state(W,N=1_200_000,CH=60_000):
    """true post-relu mean m^l and second moment DIAGONAL, plus true full K diag-anchor pieces."""
    s1=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]
    s2=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]
    SK=[torch.zeros(D,D,dtype=torch.float64,device=dev) for _ in range(L)]
    n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(12000+c))
        x=torch.cat([z,-z],0)
        for li in range(L):
            x=relu(x@W[li]); xd=x.double()
            s1[li]+=xd.sum(0); s2[li]+=(xd*xd).sum(0); SK[li]+=xd.t()@xd
        n+=x.shape[0]
    return [s/n for s in s1],[s/n for s in s2],[s/n for s in SK]

def readout(m,K,Wl):
    mu=Wl.t()@m; C=Wl.t()@(K-torch.outer(m,m))@Wl
    sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-12)); t=mu/sd; Ph=Phi(t); ph=phi(t)
    mn=mu*Ph+sd*ph
    rho=(C/torch.outer(sd,sd)).clamp(-0.999999,0.999999); sq=torch.sqrt(1-rho*rho)
    Kg=torch.zeros(D,D,device=dev); sdsq=sd[None,:]*sq
    for a in range(H):
        Ri=relu(mu+sd*xn[a]); gjb=mu[None,:]+sd[None,:]*(rho*xn[a])
        Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*xn[:,None,None])
        Kg+=wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
    Kg[range(D),range(D)]=(mu*mu+sd*sd)*Ph+mu*sd*ph
    return mn,Kg

def propagate(W,mt,kt,KT,every=None,mode='full'):
    """every=None -> pure NNGP.  else re-anchor with ORACLE state every k layers."""
    m=torch.zeros(D,device=dev); K=torch.eye(D,device=dev); tr=[]
    for ell in range(L):
        m,K=readout(m,K,W[ell])
        if every is not None and ell%every==every-1 and ell<L-1:
            if mode=='full':
                m=mt[ell].float(); K=KT[ell].float()
            elif mode=='diag':                       # anchor only the DIAGONAL + mean
                Kd=K.clone(); Kd[range(D),range(D)]=kt[ell].float(); K=Kd; m=mt[ell].float()
            elif mode=='scalar':                     # anchor only the TRACE (one number)
                sc=float(kt[ell].sum())/float(torch.diagonal(K).sum())
                K=K*sc; m=m*math.sqrt(sc)
        if ell in TAPS: tr.append(float(torch.diagonal(K).sum()))
    return np.array(tr)

NN=4
log('=== RE-ANCHORING: how much of the 6.5x compounding is recoverable? ===')
log('(oracle anchors, to find the CEILING of the idea and the needed anchor density)')
log('mode   | every | RMS rel.err of Tr(K) over {} nets'.format(NN))
res={}
for net in range(NN):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    mt,kt,KT=true_state(W)
    truth=np.array([float(kt[e].sum()) for e in TAPS])
    for mode in ['full','diag','scalar']:
        for ev in [None,16,8,4,2,1]:
            if mode!='full' and ev is None: continue
            key=(mode,ev)
            e=(propagate(W,mt,kt,KT,ev,mode)-truth)/truth
            res.setdefault(key,[]).append(e)
    log('  net {} done'.format(net))
log('')
base=None
for (mode,ev),v in sorted(res.items(), key=lambda kv:(kv[0][0],kv[0][1] if kv[0][1] else 99)):
    r=np.sqrt((np.array(v)**2).mean())
    if ev is None: base=r
    log('{:7s}| {:5s} | {:.4e}   {}'.format(mode,str(ev),r,
        '' if base is None else '({:.2f}x better than pure NNGP)'.format(base/r)))
log('')
log('TARGET: need 1.4e-4 (7.6x better than the ~1e-3 of pure NNGP)')
log('one-step closure error floor measured earlier: 1.55e-4')
log('done.')
