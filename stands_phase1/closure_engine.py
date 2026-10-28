import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG='/mnt/win_d/llm/White-Box/closure_engine.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:2'; D=256; L=32; H=24
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
xh,wh=hermgauss(H)
xn=torch.tensor(xh*SQ2,dtype=torch.float32,device=dev)
wn=torch.tensor(wh/math.sqrt(math.pi),dtype=torch.float32,device=dev)

d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
log('net0 loaded, sig~{:.4f}'.format(float(W[0].std())))

# ---------- Gaussian readouts ----------
def phi(t): return torch.exp(-0.5*t*t)/SQ2PI
def Phi(t): return 0.5*(1+torch.erf(t/SQ2))
def readout(m_prev,K_prev,Wl):
    mu=Wl.t()@m_prev
    C=K_prev-torch.outer(m_prev,m_prev)
    Cz=Wl.t()@C@Wl
    sd=torch.sqrt(torch.diagonal(Cz).clamp(min=1e-12))
    t=mu/sd; Ph=Phi(t); ph=phi(t)
    m_g=mu*Ph+sd*ph
    # off-diagonal second moment via 2D Gauss-Hermite
    rho=(Cz/torch.outer(sd,sd)).clamp(-0.999999,0.999999); sq=torch.sqrt(1-rho*rho)
    Kg=torch.zeros(D,D,device=dev); sdsq=sd[None,:]*sq
    for a in range(H):
        Ri=relu(mu+sd*xn[a])
        gjb=mu[None,:]+sd[None,:]*(rho*xn[a])
        Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*xn[:,None,None])
        Kg+=wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
    # exact diagonal E[relu^2] = (mu^2+sd^2)Phi + mu*sd*phi
    diag=(mu*mu+sd*sd)*Ph+mu*sd*ph
    Kg[range(D),range(D)]=diag
    return mu,Cz,m_g,Kg

# ---------- true per-layer moments (MC, antithetic) ----------
def collect_true(N=8_000_000,CH=50_000):
    npair=N//2; nch=npair//CH
    s1=[torch.zeros(D,device=dev) for _ in range(L)]
    S2=[torch.zeros(D,D,device=dev) for _ in range(L)]
    for c in range(nch):
        z0=torch.randn(CH,D,device=dev,generator=torch.Generator(device=dev).manual_seed(7000+c))
        x=torch.cat([z0,-z0],0)
        for li in range(L):
            x=relu(x@W[li])
            s1[li]+=x.sum(0); S2[li]+=x.t()@x
    Nt=nch*CH*2
    m_true=[s/Nt for s in s1]; K_true=[s/Nt for s in S2]
    return m_true,K_true,Nt

log('collecting true per-layer moments (N=8M antithetic)...')
m_true,K_true,Nt=collect_true()
log('collected, Nt={}'.format(Nt))

# ---------- oracle per-layer closure corrections ----------
# layer 0: z^0 exactly Gaussian (linear in Gaussian x) -> readout exact, no correction
m0,K0=torch.zeros(D,device=dev),torch.eye(D,device=dev)
_,_,mg0,Kg0=readout(m0,K0,W[0])
dm=[None]*L; dK=[None]*L   # index by layer; dm[0]=dK[0]=0
dm[0]=torch.zeros(D,device=dev); dK[0]=torch.zeros(D,D,device=dev)
for ell in range(1,L):
    _,_,mg,Kg=readout(m_true[ell-1],K_true[ell-1],W[ell])
    dm[ell]=m_true[ell]-mg
    dK[ell]=K_true[ell]-Kg

# rank-k truncation of dK (symmetric, top-|eig|), done on CPU
def rankk(M,k):
    if k>=D: return M
    Ms=0.5*(M+M.t()).cpu()
    ev,V=torch.linalg.eigh(Ms)
    idx=ev.abs().sort(descending=True).indices[:k]
    Mk=(V[:,idx]*ev[idx])@V[:,idx].t()
    return Mk.to(dev)

# ---------- propagate with a given correction policy ----------
def propagate(use_dm,use_dK,krank=None):
    _,_,m,K=readout(m0,K0,W[0])       # exact layer0
    m=m+dm[0]; K=K+dK[0]
    for ell in range(1,L):
        _,_,mg,Kg=readout(m,K,W[ell])
        m=mg+(dm[ell] if use_dm else 0.0)
        if use_dK:
            corr=dK[ell] if krank is None else rankk(dK[ell],krank)
            K=Kg+corr
        else:
            K=Kg
    return m

tgt=m_true[L-1]
def mse(v): return float(((v-tgt)**2).mean())
base=mse(propagate(False,False))
dm_only=mse(propagate(True,False))
dK_only=mse(propagate(False,True))
full=mse(propagate(True,True))
log('=== FINAL MEAN MSE vs true E[a^L] (net0) ===')
log('NNGP baseline (no corr)      : {:.4e}   (v2={:.2e})'.format(base,3.09e-7))
log('+ dm only (mean closure)     : {:.4e}   ratio {:.3f}'.format(dm_only,dm_only/base))
log('+ dK only (2nd-moment closure): {:.4e}  ratio {:.3f}'.format(dK_only,dK_only/base))
log('+ dm + dK FULL (oracle ceiling): {:.4e} ratio {:.3f}'.format(full,full/base))
log('--- rank-truncated dK (with full dm) : is rank-8 enough? ---')
for k in [2,4,8,16,32,64]:
    v=mse(propagate(True,True,krank=k))
    log('  dm + dK(rank {:>2})            : {:.4e}   ratio {:.3f}'.format(k,v,v/base))
# sanity: MC noise floor of the target itself ~ how well can we even measure
log('note: target m_true from N={} antithetic; its own MSE noise ~ few e-9'.format(Nt))
log('done.')
