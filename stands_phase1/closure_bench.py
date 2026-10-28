"""ШВИДКИЙ СТЕНД ЗАМИКАННЯ (k=2): чи можна полагодити перший крок?

crossover.py: увесь провал аналітики народжується на переході 1-й→2-й шар (1.7e-5),
а всі 30 наступних шарів додають лише ×9.

При k=2 ситуація ІДЕАЛЬНА для діагностики:
  - вхід x рівно гаусів
  - a¹=relu(xW₀): m¹ і C¹ обчислюються ТОЧНО (зрізана гаусіана)
  - отже μ=W₁ᵀm¹ і σ²=diag(W₁ᵀC¹W₁) для h²=a¹W₁ теж ТОЧНІ
  ⟹ ЄДИНА похибка = НЕГАУСОВІСТЬ h². Більше нічого.

Значить її має виправляти поправка Еджворта:
  E[f(h)] = E[f(g)] + (κ₃/6)E[f'''(g)] + (κ₄/24)E[f⁗(g)] + ...
  f=relu:  f'''=δ',  f⁗=δ''   ->  для g~N(μ,σ²), t=μ/σ:
     Δ₃ = (κ₃/6)·(−t·φ(t)/σ²)
     Δ₄ = (κ₄/24)·((t²−1)·φ(t)/σ³)

⚠️ ОРАКУЛ: κ₃,κ₄ беру з MC (істинні). Це міряє СТЕЛЮ методу, не сам метод.
Якщо навіть з оракулом похибка не падає — Gram-Charlier мертвий у самому джерелі.
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/closure_bench.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; DT=torch.float64
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
phi=lambda t: torch.exp(-0.5*t*t)/SQ2PI
Phi=lambda t: 0.5*(1+torch.erf(t/SQ2))

for net in [0,1,2]:
    W0=torch.tensor(dat['w_%d_0'%net].astype(np.float32),device=dev).double()
    W1=torch.tensor(dat['w_%d_1'%net].astype(np.float32),device=dev).double()
    log(''); log('######## НЕТ {} ########'.format(net))

    # --- ТОЧНІ m¹, C¹ для a¹=relu(xW₀), x~N(0,I): h¹~N(0, W₀ᵀW₀) ---
    C1h=W0.t()@W0
    s1=torch.sqrt(torch.diagonal(C1h).clamp(min=1e-14))
    m1=s1/SQ2PI                                   # E[relu(g)] при μ=0 -> σ/√(2π)
    rho=(C1h/torch.outer(s1,s1)).clamp(-0.999999,0.999999)
    # E[relu(g_i)relu(g_j)] для центрованої пари (закрита формула)
    ang=torch.acos(rho)
    E2=(torch.outer(s1,s1)/(2*math.pi))*(torch.sqrt((1-rho*rho).clamp(min=0))+(math.pi-ang)*rho)
    E2=E2-torch.diag(torch.diagonal(E2))+torch.diag(s1*s1*0.5)     # діагональ: E[relu²]=σ²/2
    C1=E2-torch.outer(m1,m1)                      # ТОЧНА коваріація a¹

    mu=W1.t()@m1
    var=torch.diagonal(W1.t()@C1@W1).clamp(min=1e-14)
    sig=var.sqrt(); t=mu/sig
    nngp_mean=mu*Phi(t)+sig*phi(t)                # гаусове замикання

    # --- істина + ОРАКУЛЬНІ кумулянти h² ---
    NT=8_000_000; CH=200_000
    s_a=torch.zeros(D,dtype=DT,device=dev)
    c1_=torch.zeros(D,dtype=DT,device=dev); c2_=torch.zeros(D,dtype=DT,device=dev)
    c3_=torch.zeros(D,dtype=DT,device=dev); c4_=torch.zeros(D,dtype=DT,device=dev); n=0
    for c in range(NT//CH):
        g=torch.Generator(device=dev).manual_seed(9900+net*11+c)
        z=torch.randn(CH//2,D,device=dev,generator=g,dtype=DT); x=torch.cat([z,-z],0)
        h2=relu(x@W0)@W1
        d=h2-mu[None]
        s_a+=relu(h2).sum(0)
        c1_+=d.sum(0); c2_+=(d*d).sum(0); c3_+=(d**3).sum(0); c4_+=(d**4).sum(0)
        n+=x.shape[0]
    TH=s_a/n
    M1=c1_/n; M2=c2_/n-M1*M1; M3=c3_/n-3*M1*(c2_/n)+2*M1**3
    M4=c4_/n-4*M1*(c3_/n)+6*M1*M1*(c2_/n)-3*M1**4
    k3=M3; k4=M4-3*M2*M2                          # кумулянти h²

    p=float((TH*TH).mean())
    def rel(e): return float(((e-TH)**2).mean())/p
    d3=(k3/6.0)*(-t*phi(t)/var)
    d4=(k4/24.0)*((t*t-1)*phi(t)/(sig*var))
    log('  перевірка: σ² NNGP проти MC — відн.розб {:.2e} (має бути ~0, μ,σ ТОЧНІ)'.format(
        float(((var-M2)/M2).abs().mean())))
    log('  скіс γ₁ сер {:+.4f}   куртозис γ₂ сер {:+.4f}'.format(
        float((k3/M2.pow(1.5)).mean()), float((k4/(M2*M2)).mean())))
    log('')
    log('  гаусове замикання (NNGP)     відн.MSE = {:.4e}'.format(rel(nngp_mean)))
    log('  + κ₃ (ОРАКУЛ)                відн.MSE = {:.4e}   ({:.2f}× краще)'.format(
        rel(nngp_mean+d3), rel(nngp_mean)/max(rel(nngp_mean+d3),1e-300)))
    log('  + κ₃ + κ₄ (ОРАКУЛ)           відн.MSE = {:.4e}   ({:.2f}× краще)'.format(
        rel(nngp_mean+d3+d4), rel(nngp_mean)/max(rel(nngp_mean+d3+d4),1e-300)))
    log('  ОРІЄНТИР: v2 при k=2 дає ~7.2e-7 (crossover.py)')
log('done.')
