"""НАЧИСТО: розщеплення вибірки + усадка + змішаний якір.

Оцінювач:  theta_j = f_j(N1) - lambda * beta_j^T ( T(N1) - mu_hat )
Якір:      mu_hat  = w * mu_NNGP  +  (1-w) * mu_MC(N2)      N1+N2 = N
NNGP безкоштовний але зміщений (delta), MC незміщений але коштує семпли.

MSE_j = s_j^2/N1 * (1 - 2*l*R2_j + l^2*R2_j)          дисперсія основного + CV
      + l^2 (1-w)^2 R2_j s_j^2 / N2                    шум MC-якоря
      + l^2 w^2 (beta_j^T delta)^2                     зсув NNGP-якоря

Усі три кути — окремі випадки: (w=1,N2=0) сьогоднішня усадка; (w=0) чистий MC-якір;
(l=1) те, що я рахував як "програш x19".
delta береться з ПРІОРУ по банку 200 нетів (чесно, не з цього ж нета).
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/split.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
TAPS=[3,7,11,15,19,23,27,30]; k=len(TAPS); V2=3.09e-7

# --- пріор на похибку слідів NNGP (200 нетів, ВІДНОСНА) ---
c=torch.load('/mnt/win_d/llm/White-Box/netbank/micro_trace_cache.pt',weights_only=False)
REL=(np.asarray(c['TRN'])-np.asarray(c['TRT']))/np.asarray(c['TRT'])
SIG_REL=(REL.T@REL)/len(REL)            # E[delta_rel delta_rel^T]  (8x8)
log('пріор NNGP: RMS відносної похибки слідів = {:.3e}'.format(np.sqrt(np.diag(SIG_REL).mean())))

def moments(W,seed0,nch,CH=10000):
    A=torch.zeros(k,k,dtype=torch.float64,device=dev); BY=torch.zeros(k,D,dtype=torch.float64,device=dev)
    SX=torch.zeros(k,dtype=torch.float64,device=dev)
    SY=torch.zeros(D,dtype=torch.float64,device=dev); SY2=torch.zeros(D,dtype=torch.float64,device=dev)
    N=0
    def fwd(x):
        out=[]
        for li in range(L):
            x=relu(x@W[li])
            if li in TAPS: out.append((x*x).sum(1))
        return x,out
    for cc in range(nch):
        w=torch.randn(CH,D,device=dev,generator=torch.Generator(device=dev).manual_seed(seed0+cc))
        ap,na=fwd(w); am,nb=fwd(-w)
        Y=(0.5*(ap+am)).double(); T=torch.stack([0.5*(a+b) for a,b in zip(na,nb)],1).double()
        A+=T.t()@T; BY+=T.t()@Y; SX+=T.sum(0); SY+=Y.sum(0); SY2+=(Y*Y).sum(0); N+=CH
    return [t.cpu() for t in (A,BY,SX,SY,SY2)]+[N]

log('')
for net in [0,1,2]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    A,BY,SX,SY,SY2,N=moments(W,60000+net*997,40)
    Ah,BYh,SXh,SYh,SY2h,Nh=moments(W,90000+net*997,20)
    SStot=SY2-SY*SY/N; alive=(SStot>1e-9*float(SStot.max())).numpy()
    S=(A-torch.outer(SX,SX)/N)/N                       # Cov(T)  (8x8)
    Cf=(BY-torch.outer(SX,SY)/N)/N                     # Cov(T,f) (8xD)
    beta=torch.linalg.solve(S+1e-9*torch.eye(k,dtype=torch.float64),Cf)   # (8,D)
    # held-out R^2 покомпонентно
    SStoth=SY2h-SYh*SYh/Nh
    Ach=(Ah-torch.outer(SXh,SXh)/Nh); Bch=(BYh-torch.outer(SXh,SYh)/Nh)
    SSresh=SStoth-2*(beta*Bch).sum(0)+((beta.t()@Ach)*beta.t()).sum(1)
    R2=(1-SSresh/SStoth).clamp(0,0.999).numpy()
    Etr=(SX/N).numpy()                                  # масштаб слідів
    SIG=SIG_REL*np.outer(Etr,Etr)                       # абсолютна коваріація delta
    b2=np.einsum('kj,kl,lj->j',beta.numpy(),SIG,beta.numpy())   # (beta^T delta)^2 очікуване
    sig2=(SStot/N).numpy(); sig2=sig2/sig2[alive].mean()*V2     # s_j^2/N нормовані на v2

    def mse(l,w,frac):
        n1,n2=frac,1-frac
        if n2<=0 and w<1: return np.inf
        t=sig2/n1*(1-2*l*R2+l*l*R2) + (l*l*w*w)*b2
        if w<1: t=t+ (l*l*(1-w)**2*R2*sig2)/max(n2,1e-9)
        return t[alive].mean()
    best=(np.inf,)
    for l in np.linspace(0,1,101):
        for w in np.linspace(0,1,51):
            for fr in np.linspace(0.5,1.0,51):
                if w==1.0 and fr<1.0: continue
                v=mse(l,w,fr)
                if v<best[0]: best=(v,l,w,fr)
    log('NET {}: R2_ho сер {:.4f}'.format(net,R2[alive].mean()))
    log('   без CV (=v2)                      {:.4e}'.format(V2))
    m1=min(mse(l,1.0,1.0) for l in np.linspace(0,1,101))
    log('   якір NNGP, усадка (сьогодні)      {:.4e}  ({:+.2f}%)'.format(m1,100*(m1/V2-1)))
    m2=min(mse(l,0.0,fr) for l in np.linspace(0,1,101) for fr in np.linspace(0.5,0.99,50))
    log('   якір MC, розщеплення+усадка       {:.4e}  ({:+.2f}%)'.format(m2,100*(m2/V2-1)))
    m3=mse(1.0,0.0,0.5)
    log('   якір MC, 50/50, БЕЗ усадки        {:.4e}  ({:+.2f}%)'.format(m3,100*(m3/V2-1)))
    log('   ЗМІШАНИЙ оптимум                  {:.4e}  ({:+.2f}%)   l={:.2f} w={:.2f} N1={:.0f}%'.format(
        best[0],100*(best[0]/V2-1),best[1],best[2],100*best[3]))
    mi=min(sig2[alive].mean()*(1-r) for r in [R2[alive].mean()])
    log('   стеля (ідеальний безкоштовний якір) {:.4e}  ({:+.2f}%)'.format(
        (sig2*(1-R2))[alive].mean(),100*((sig2*(1-R2))[alive].mean()/V2-1)))
log('done.')
