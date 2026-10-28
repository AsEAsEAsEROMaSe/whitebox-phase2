"""ЗАКОН АРКСИНУСА як ТОЧНИЙ БЕЗКОШТОВНИЙ ЯКІР (ідея юзера «рівні переходу», С47).

ЛОГІКА:
  s_i = 1[h_i>0] на ПЕРШОМУ шарі, h = x@W0, x~N(0,I) => h ~ N(0,Sig), Sig = W0^T W0.
  u_i = s_i - 1/2 in {+-1/2}.  При x->-x  u -> -u  =>  u_i u_j ПАРНЕ (виживає антитетику),
  лінійне u_i НЕПАРНЕ (має дати R^2~0 — вбудований контроль).
  E[u_i u_j] = arcsin(rho_ij)/(2pi) ТОЧНО (діагональ: arcsin(1)/2pi = 1/4). Ціна d^2 = БЕЗКОШТОВНО.
  Якір НЕ КУПУЄТЬСЯ => тотожність вартості якоря не спрацьовує.

ХІД: A = arcsin(rho)/(2pi); власні вектори V (топ-M) => p = u@V;  E[p_a p_b] = lam_a*delta_ab.
     Ознаки f_ab = p_a p_b - lam_a*delta_ab (a<=b), середнє ТОЧНО нуль.
"""
import torch, time, numpy as np, math
LOG='/mnt/win_d/llm/White-Box/arcsine_cv.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; M=24            # M мод -> M(M+1)/2 = 300 ознак
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
iu=torch.triu_indices(M,M,device=dev)

def run(net):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    W0d=W[0].double(); Sig=W0d.t()@W0d; dg=Sig.diag().rsqrt()
    rho=(Sig*dg[:,None])*dg[None,:]
    A=torch.arcsin(rho.clamp(-1,1))/(2*math.pi)                 # ТОЧНЕ E[u u^T]
    ev,V=torch.linalg.eigh(A.cpu())
    lam=ev.flip(0)[:M].to(dev); V=V.flip(1)[:,:M].to(dev); Vf=V.float()
    log('  власні числа арксинус-матриці: {:.4f}..{:.4f}; слід={:.3f} (теорія d/4={:.1f})'
        .format(float(lam[0]),float(lam[-1]),float(A.diag().sum()),D/4))
    anc=torch.where(iu[0]==iu[1],lam[iu[0]],torch.zeros(iu.shape[1],dtype=lam.dtype,device=dev))

    NP=400_000; CH=25_000
    Fq=[]; Fl=[]; Y=[]
    for c in range(NP//CH):
        gg=torch.Generator(device=dev).manual_seed(4400+c)
        z=torch.randn(CH,D,device=dev,generator=gg)
        x=torch.cat([z,-z],0)
        h0=x@W[0]; u=(h0>0).float()-0.5
        p=(u@Vf).double(); pz=p[:CH]
        if c==0:
            log('  КОНТРОЛЬ парності: max|p(x)+p(-x)| = {:.2e} (0 => u строго непарне)'
                .format(float((pz+p[CH:]).abs().max())))
        Fq.append(pz[:,iu[0]]*pz[:,iu[1]]-anc)                  # ПАРНІ, середнє точно 0
        Fl.append(pz)                                            # НЕПАРНІ — контроль
        a=torch.relu(h0)
        for li in range(1,L): a=torch.relu(a@W[li])
        Y.append(((a[:CH]+a[CH:])*0.5).double())                 # парна частина = ціль v2
    Fq=torch.cat(Fq); Fl=torch.cat(Fl); Y=torch.cat(Y); N=Y.shape[0]
    log('  |емпіричне середнє ознак|max = {:.3e}   (3-сигма шуму = {:.3e})'
        .format(float(Fq.mean(0).abs().max()), 3*float((Fq.std(0)/N**0.5).max())))

    h=N//2; Yc=Y-Y[:h].mean(0)
    def fit(Fx,name):
        sd=Fx[:h].std(0).clamp(min=1e-12); Fs=Fx/sd
        Aa=Fs[:h]; G=Aa.t()@Aa/h; rhs=Aa.t()@Yc[:h]/h; I=torch.eye(G.shape[0],dtype=G.dtype,device=dev)
        best=(None,-9.9)
        for lm in (1e-6,1e-4,1e-2,1e-1):
            b=torch.linalg.solve(G+lm*I,rhs); r=Yc[h:]-Fs[h:]@b
            r2=1-float((r*r).sum()/(Yc[h:]**2).sum())
            if r2>best[1]: best=(lm,r2)
        log('  {:24s} ознак={:5d}  R^2(відкладена)={:+.4f}  (lam={:.0e})'
            .format(name,Fx.shape[1],best[1],best[0]))
        return best[1]
    fit(Fl,'ЛІНІЙНІ (контроль ~0)')
    r2q=fit(Fq,'КВАДРАТИЧНІ (арксинус)')
    fit(torch.cat([Fl,Fq],1),'разом')
    log('  => зниження MSE оцінювача: {:.1f}%'.format(100*max(r2q,0)))

for net in (0,1,2):
    log('==== НЕТ {} ===='.format(net)); run(net)
log('done.')
