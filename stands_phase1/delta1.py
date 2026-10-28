"""ПЕРШИЙ ПОРЯДОК δ-РОЗКЛАДУ — чи він великий і чи береться аналітично.

  a^L = Π_l (W_l G_l) застосоване до x,   G_l = ½I + δ_l = diag(g_l)
      = Π_l (A_l + W_l δ_l),              A_l = ½W_l
  S=∅      -> 2^{-L}·x·ΠW  -> E=0 ТОЧНО (підтверджено halfgate.py: 0.0018 еталона)
  S={l}    -> T_l = 2^{-(L-1)} · E[u^l ⊙ δ_l] · Q_l
             u^l = x·W_1···W_l  = вихід ЛІНІЙНОЇ мережі (рівно гаусів!)
             Q_l = W_{l+1}···W_L
             δ_l = diag(g_l − ½)

ЧОМУ ЦЕ МОЖЕ ДАТИ БЕЗКОШТОВНИЙ ЯКІР:
  E[u ⊙ δ] = E[u·1(h>0)] − ½E[u] = E[u·1(h>0)]   (бо E[u]=0)
  для СПІЛЬНО ГАУСОВОЇ пари (u,h):  E[u·1(h>0)] = Cov(u,h)/σ_h · φ(0)
  σ_h -> діагональ NNGP (є);  Cov(u,h) -> через середній якобіан (L матмулів)
  ⟹ увесь перший порядок = ЗАКРИТА ФОРМУЛА, великі матмули, копійки флопів

ЦЕЙ ПРОГІН міряє MC-версію: наскільки Σ T_l пояснює E[a^L].
Якщо частка мала — дорога мертва одразу, аналітику писати нема сенсу.
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/delta1.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
SQ2PI=math.sqrt(2*math.pi)

N=400_000; CH=50_000
for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    Wd=[w.double() for w in W]
    log(''); log('######## НЕТ {}  (N={}) ########'.format(net,N))

    # ---- MC: E[a^L], а також v_l = E[u^l ⊙ δ_l] і Cov(u^l,h^l), Var(h^l) ----
    sA=torch.zeros(D,dtype=torch.float64,device=dev)
    V =[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]   # E[u ⊙ δ]
    CUH=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]  # E[u*h]
    VH =[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]  # E[h^2]
    n=0
    for c in range(N//CH):
        g=torch.Generator(device=dev).manual_seed(9100+net*13+c)
        z=torch.randn(CH//2,D,device=dev,generator=g)
        x=torch.cat([z,-z],0)
        a=x; u=x
        for li in range(L):
            h=a@W[li]; u=u@W[li]              # u = ЛІНІЙНА мережа (без relu)
            gt=(h>0).to(h.dtype)
            ud=u.double(); hd=h.double()
            V[li]  += (ud*(gt.double()-0.5)).sum(0)
            CUH[li]+= (ud*hd).sum(0)
            VH[li] += (hd*hd).sum(0)
            a=h*gt
        sA+=a.double().sum(0); n+=x.shape[0]
    base=sA/n
    V=[v/n for v in V]; CUH=[c_/n for c_ in CUH]; VH=[v/n for v in VH]
    nb=float(base.norm())
    log('  ||E[a^L]|| = {:.6f}'.format(nb))

    # ---- суфіксні добутки Q_l = W_{l+1}···W_{L-1} ----
    Q=[None]*L
    Q[L-1]=torch.eye(D,dtype=torch.float64,device=dev)
    for li in range(L-2,-1,-1): Q[li]=Wd[li+1]@Q[li+1]

    pref=2.0**(-(L-1))
    T=[pref*(V[li]@Q[li]) for li in range(L)]
    S1=torch.stack(T).sum(0)
    log('')
    log('  ПЕРШИЙ ПОРЯДОК (MC-версія):')
    log('    ||Σ T_l||        = {:.6e}   це {:.4f} від ||E[a^L]||'.format(
        float(S1.norm()), float(S1.norm())/nb))
    log('    ||E[a^L] − ΣT_l|| = {:.6e}   залишок {:.4f}'.format(
        float((base-S1).norm()), float((base-S1).norm())/nb))
    cs=float((base@S1)/(base.norm()*S1.norm().clamp(min=1e-300)))
    log('    cos(E[a^L], ΣT_l) = {:+.4f}'.format(cs))
    log('    внесок окремих шарів ||T_l||/||E||:')
    rr=[float(t.norm())/nb for t in T]
    log('      шари 0-7 : ' + ' '.join('{:.3f}'.format(v) for v in rr[:8]))
    log('      шари 8-15: ' + ' '.join('{:.3f}'.format(v) for v in rr[8:16]))
    log('      шари16-23: ' + ' '.join('{:.3f}'.format(v) for v in rr[16:24]))
    log('      шари24-31: ' + ' '.join('{:.3f}'.format(v) for v in rr[24:]))

    # ---- АНАЛІТИЧНА версія v_l через гаусову формулу (перевірка придатності) ----
    log('')
    log('  ЧИ БЕРЕТЬСЯ v_l = E[u⊙δ] ГАУСОВОЮ ФОРМУЛОЮ Cov(u,h)/σ_h·φ(0)?')
    log('    шар   ||v_ан−v_MC||/||v_MC||   cos')
    for li in [0,1,3,7,15,23,29,31]:
        sh=VH[li].clamp(min=1e-30).sqrt()
        van=CUH[li]/sh/SQ2PI                    # φ(0)=1/√(2π)
        rel=float((van-V[li]).norm()/V[li].norm().clamp(min=1e-300))
        cc=float((van@V[li])/(van.norm()*V[li].norm()).clamp(min=1e-300))
        log('    {:2d}      {:.4f}                 {:+.4f}'.format(li,rel,cc))
log('done.')
