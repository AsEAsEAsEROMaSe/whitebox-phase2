import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG='/mnt/win_d/llm/White-Box/probe_closure_rank.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:2'; D=256; L=32; H=24
relu=lambda x: torch.clamp(x,min=0.0)
xh,wh=hermgauss(H)
xn=torch.tensor(xh*math.sqrt(2),dtype=torch.float32,device=dev)
wn=torch.tensor(wh/math.sqrt(math.pi),dtype=torch.float32,device=dev)

d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
log('net0 loaded, W[0] {} sig~{:.4f}'.format(tuple(W[0].shape), float(W[0].std())))

def gauss_secondmoment(mu,C):
    sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-12))
    rho=(C/torch.outer(sd,sd)).clamp(-0.999999,0.999999); sq=torch.sqrt(1-rho*rho)
    M=torch.zeros(D,D,device=dev); sdsq=sd[None,:]*sq
    for a in range(H):
        Ri=relu(mu+sd*xn[a])
        gjb=mu[None,:]+sd[None,:]*(rho*xn[a])
        Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*xn[:,None,None])
        M+=wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
    return M

def probe_layer(ell,N=2_000_000,CH=250_000):
    nch=N//CH
    s1=torch.zeros(D,device=dev); Szz=torch.zeros(D,D,device=dev)
    Mrr=torch.zeros(D,D,device=dev); r1=torch.zeros(D,device=dev)
    for c in range(nch):
        z0=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(1000+c))
        x=torch.cat([z0,-z0],0); zc=None
        for li in range(ell):
            h=x@W[li]
            if li<ell-1: x=relu(h)
            else: zc=h
        s1+=zc.sum(0); Szz+=zc.t()@zc
        a=relu(zc); r1+=a.sum(0); Mrr+=a.t()@a
    Nt=nch*CH
    mu=s1/Nt; C=Szz/Nt-torch.outer(mu,mu)
    M_true=Mrr/Nt; M_g=gauss_secondmoment(mu,C); dM=M_true-M_g
    dM_cpu=(0.5*(dM+dM.t())).cpu()
    evD,_=torch.linalg.eigh(dM_cpu); evD=evD.to(dev)
    aD=evD.abs().sort(descending=True).values
    pr=(aD.sum()**2/(aD**2).sum()).item()
    fro_true=torch.linalg.norm(M_true).item(); fro_d=torch.linalg.norm(dM).item()
    top=aD[:8]/aD[0]
    energy=lambda k:(aD[:k].sum()/aD.sum()).item()
    log('L{:>2}: |dM|/|M|={:.4f}  PR(dM)={:5.1f}/256  E(top5)={:.3f} E(top10)={:.3f} E(top20)={:.3f}'.format(
        ell,fro_d/fro_true,pr,energy(5),energy(10),energy(20)))
    log('      top8 |eig(dM)| ratios: '+' '.join('{:.3f}'.format(v.item()) for v in top))
    return pr

log('=== closure error rank: is finite-width correction low-rank? (net0, fp32, N=2M/layer) ===')
log('measure dM = M_true(relu z) - M_gauss(relu z); PR small => poly path to finite-width correction')
for ell in [4,8,16,24,31]:
    probe_layer(ell)
log('done.')
