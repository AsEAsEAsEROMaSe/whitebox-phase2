import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/strat_probe.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; NBIN=256
relu=lambda x: torch.clamp(x,min=0.0)
d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
G0=W[0].t()@W[0]; std0=torch.sqrt(torch.diagonal(G0).clamp(min=1e-12))

def run(w):
    """forward, return final Y (antithetic-even) and candidate scalar statistics T."""
    def fwd(x, taps):
        out={}
        for li in range(L):
            x=relu(x@W[li])
            if li in taps: out[li]=x
        return x,out
    taps={7,15,23,30}
    ap,ta=fwd(w,taps); am,tb=fwd(-w,taps)
    Y=0.5*(ap+am)
    s=(w@W[0])/std0
    T={}
    T['input ||x||^2 (radial)']      = (w*w).sum(1)
    T['input strongest channel s_i'] = s[:,0]
    T['layer8  ||a||^2']  = 0.5*((ta[7]**2).sum(1)+(tb[7]**2).sum(1))
    T['layer16 ||a||^2']  = 0.5*((ta[15]**2).sum(1)+(tb[15]**2).sum(1))
    T['layer24 ||a||^2']  = 0.5*((ta[23]**2).sum(1)+(tb[23]**2).sum(1))
    T['layer31 ||a||^2']  = 0.5*((ta[30]**2).sum(1)+(tb[30]**2).sum(1))
    T['layer31 neuron0']  = 0.5*(ta[30][:,0]+tb[30][:,0])
    return Y,T

# accumulate per-stratum sums to get Var(E[Y|T]) / Var(Y)
NB=400_000; CH=10_000; nch=NB//CH
names=None; edges={};
log('pass1: quantile edges for each statistic ...')
w=torch.randn(40_000,D,device=dev,generator=torch.Generator(device=dev).manual_seed(31))
Y0,T0d=run(w); names=list(T0d.keys())
for k in names:
    q=torch.quantile(T0d[k].float(), torch.linspace(0,1,NBIN+1,device=dev))
    q[0]=-1e30; q[-1]=1e30; edges[k]=q.contiguous()
log('pass2: accumulating strata over {} samples ...'.format(NB))
cnt={k:torch.zeros(NBIN,dtype=torch.float64,device=dev) for k in names}
sY ={k:torch.zeros(NBIN,D,dtype=torch.float64,device=dev) for k in names}
SY=torch.zeros(D,dtype=torch.float64,device=dev); SY2=torch.zeros(D,dtype=torch.float64,device=dev); N=0
for c in range(nch):
    w=torch.randn(CH,D,device=dev,generator=torch.Generator(device=dev).manual_seed(700+c))
    Y,T=run(w); Yd=Y.double()
    for k in names:
        b=(torch.bucketize(T[k].float(),edges[k])-1).clamp(0,NBIN-1)
        cnt[k].index_add_(0,b,torch.ones(CH,dtype=torch.float64,device=dev))
        sY[k].index_add_(0,b,Yd)
    SY+=Yd.sum(0); SY2+=(Yd*Yd).sum(0); N+=CH
SStot=SY2-SY*SY/N
alive=SStot>1e-9*float(SStot.max()); nA=int(alive.sum())
log('=== STRATIFICATION PRIZE: Var(E[Y|T]) / Var(Y), {} strata, N={} ==='.format(NBIN,N))
log('alive {}/{}   (this is the CEILING of perfect stratification on T)'.format(nA,D))
for k in names:
    c=cnt[k].clamp(min=1); mu=sY[k]/c[:,None]
    between=(c[:,None]*(mu-(SY/N)[None,:])**2).sum(0)     # (D,)
    frac=(between/SStot)[alive].mean()
    log('  {:26s} : {:.4f}'.format(k,float(frac)))
log('reference: whole Hermite CV family on input = 0.0104 (1.04%)')
log('done.')
