import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG='/mnt/win_d/llm/White-Box/sign_bracket.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; H=40
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
xh,wh=hermgauss(H)
gx=torch.tensor(xh*SQ2,dtype=torch.float64,device=dev)      # nodes for N(0,1)
gw=torch.tensor(wh/math.sqrt(math.pi),dtype=torch.float64,device=dev)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
TAPS=[3,7,11,15,19,23,27,30]
He4=lambda u: u**4-6*u*u+3
He3=lambda u: u**3-3*u

def true_layer_stats(W,N=1_500_000,CH=75_000):
    """per-layer preactivation z^l: mean, var, skew, exkurt (per neuron) + true E[relu(z)^2] sum."""
    S1=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]
    S2=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]
    S3=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]
    S4=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]
    R2=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]
    n=0
    for c in range(N//CH):
        z0=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(11000+c))
        x=torch.cat([z0,-z0],0)
        for li in range(L):
            h=x@W[li]; hd=h.double()
            S1[li]+=hd.sum(0); S2[li]+=(hd*hd).sum(0); S3[li]+=(hd**3).sum(0); S4[li]+=(hd**4).sum(0)
            a=relu(h); R2[li]+=(a.double()**2).sum(0)
            x=a
        n+=x.shape[0]
    out=[]
    for li in range(L):
        e1=S1[li]/n; e2=S2[li]/n; e3=S3[li]/n; e4=S4[li]/n
        var=(e2-e1*e1).clamp(min=1e-12); sd=var.sqrt()
        m3=e3-3*e1*e2+2*e1**3; m4=e4-4*e1*e3+6*e1*e1*e2-3*e1**4
        out.append((e1,sd,m3/sd**3,m4/var/var-3.0,R2[li]/n))
    return out

def gauss_relu2(mu,sd):
    t=mu/sd; Ph=0.5*(1+torch.erf(t/SQ2)); ph=torch.exp(-0.5*t*t)/SQ2PI
    return (mu*mu+sd*sd)*Ph+mu*sd*ph

def edgeworth_relu2(mu,sd,skew,exk):
    """E[relu(z)^2] with Gram-Charlier skew+kurtosis correction, via 1D Gauss-Hermite."""
    u=gx[:,None]                       # (H,1) standardized nodes
    z=mu[None,:]+sd[None,:]*u
    r2=relu(z)**2
    base=(gw[:,None]*r2).sum(0)
    c3=(gw[:,None]*r2*He3(u)).sum(0)*(skew/6.0)
    c4=(gw[:,None]*r2*He4(u)).sum(0)*(exk/24.0)
    return base, base+c4, base+c3+c4

log('=== SIGN BRACKET TEST: does the kappa4-corrected closure err with OPPOSITE sign? ===')
log('one-step closure error, TRUE (mu,sd,skew,kurt) fed in, so no compounding')
log('net |    tap | gauss rel.err |  +kurt rel.err | signs opposite?')
NN=6; rows=[]
for net in range(NN):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    st=true_layer_stats(W)
    for ell in TAPS:
        mu,sd,sk,ek,truer2=st[ell]
        g=gauss_relu2(mu,sd); _,gk,gsk=edgeworth_relu2(mu,sd,sk,ek)
        Tg=float(g.sum()); Tk=float(gk.sum()); Tt=float(truer2.sum())
        eg=(Tg-Tt)/Tt; ek_=(Tk-Tt)/Tt
        rows.append((net,ell,eg,ek_))
        if ell in (3,15,30):
            log(' {:2d} |   {:2d}   |   {:+.3e}  |   {:+.3e}   | {}'.format(
                net,ell,eg,ek_,'YES' if eg*ek_<0 else 'no'))
R=np.array([(r[2],r[3]) for r in rows])
opp=(R[:,0]*R[:,1]<0).mean()
log('')
log('fraction of (net,layer) cases with OPPOSITE signs : {:.3f}  ({} cases)'.format(opp,len(R)))
log('RMS rel.err  gauss closure      : {:.4e}'.format(np.sqrt((R[:,0]**2).mean())))
log('RMS rel.err  +kurtosis          : {:.4e}'.format(np.sqrt((R[:,1]**2).mean())))
# optimal blend  w*gauss + (1-w)*kurt
ws=np.linspace(0,1,101); best=None
for w in ws:
    e=w*R[:,0]+(1-w)*R[:,1]; v=np.sqrt((e**2).mean())
    if best is None or v<best[1]: best=(w,v)
log('BEST BLEND  w={:.2f} :  RMS rel.err = {:.4e}   -> improvement over gauss {:.2f}x'.format(
    best[0],best[1],np.sqrt((R[:,0]**2).mean())/best[1]))
log('NEEDED: 7.6x')
log('done.')
