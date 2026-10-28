"""ЯК ДАЛЕКО ЙДЕ ОРБІТА + ЧИ ВИДНО T З ВАГ (продовження orbit_opt.py).

orbit_opt.py: градієнт знайшов ρ = −0.074 поза вибіркою (виграш 1.08×) і ще НЕ НАСИТИВСЯ.
⚠️ ОРАКУЛ: пошук коштував ~1e6 прогонів при бюджеті 6485. Як метод це мертво,
   ЯКЩО T треба шукати семплами (та сама [[finding-anchor-cost-identity]]).
   Рятує лише одне: ми БІЛИЙ ЯЩИК — якщо структуру T видно з ВАГ, вона коштує d³ ≈ 4 прогони.

ТРИ ПИТАННЯ:
 1. Де насичується ρ? (600 кроків замість 120)
 2. Чи має знайдена T впізнавану структуру — спектр, зв'язок із W₀?
 3. Чи переносяться дешеві кандидати з ваг: чи дає T, побудована з власних векторів
    W₀W₀ᵀ / з ядра першого шару, щось краще за випадкову?
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/orbit_opt2.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def even(W,x):
    a=x
    for w in W: a=torch.relu(a@w)
    b=-x
    for w in W: b=torch.relu(b@w)
    return (a+b)*0.5

def rho_of(W,Tm,seeds,CH=4000):
    sx=torch.zeros(D,dtype=torch.float64,device=dev); sy=torch.zeros(D,dtype=torch.float64,device=dev)
    sxx=torch.zeros(D,dtype=torch.float64,device=dev); syy=torch.zeros(D,dtype=torch.float64,device=dev)
    sxy=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for s in seeds:
        g=torch.Generator(device=dev).manual_seed(s)
        z=torch.randn(CH,D,device=dev,generator=g)
        a=even(W,z).double(); b=even(W,z@Tm).double()
        sx+=a.sum(0); sy+=b.sum(0); sxx+=(a*a).sum(0); syy+=(b*b).sum(0); sxy+=(a*b).sum(0); n+=CH
    vx=sxx/n-(sx/n)**2; vy=syy/n-(sy/n)**2; cov=sxy/n-(sx/n)*(sy/n)
    return float(cov.sum()/((vx.sum()*vy.sum())**0.5))

for net in (0,1):
    W=load(net); log('='*80); log('МЕРЕЖА %d'%net)
    g0=torch.Generator(device=dev).manual_seed(2024)
    base,_=torch.linalg.qr(torch.randn(D,D,device=dev,generator=g0))
    M=(torch.randn(D,D,device=dev,generator=g0)*0.05).requires_grad_(True)
    opt=torch.optim.Adam([M],lr=0.02)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=600)
    CH=4000
    log('  {:>6s} {:>12s} {:>14s}'.format('крок','ρ (навчання)','ρ (свіжі)'))
    for it in range(601):
        Q,R=torch.linalg.qr(base+M); Q=Q*torch.sign(torch.diagonal(R)).unsqueeze(0)
        g=torch.Generator(device=dev).manual_seed(700000+it)
        z=torch.randn(CH,D,device=dev,generator=g)
        a=even(W,z); b=even(W,z@Q)
        ac=a-a.mean(0,keepdim=True); bc=b-b.mean(0,keepdim=True)
        r=(ac*bc).sum()/torch.sqrt((ac*ac).sum()*(bc*bc).sum()+1e-30)
        opt.zero_grad(); r.backward(); opt.step(); sch.step()
        if it%100==0:
            with torch.no_grad():
                Qd,Rd=torch.linalg.qr(base+M); Qd=Qd*torch.sign(torch.diagonal(Rd)).unsqueeze(0)
                log('  {:6d} {:+12.4f} {:+14.4f}'.format(it,float(r),rho_of(W,Qd.detach(),[950001,950002,950003])))
    with torch.no_grad():
        Qd,Rd=torch.linalg.qr(base+M); Qd=Qd*torch.sign(torch.diagonal(Rd)).unsqueeze(0); Qd=Qd.detach()
        fin=rho_of(W,Qd,[960001,960002,960003,960004])
        log('  ⟹ ρ поза вибіркою {:+.4f}  ⟹ виграш {:.3f}×'.format(fin,1.0/(1.0+fin)))
        # ---- структура знайденої T ----
        ev=torch.linalg.eigvals(Qd.double()).cpu()
        ang=torch.angle(ev).abs()*180/np.pi
        h=torch.histc(ang.float(),bins=6,min=0,max=180).long().tolist()
        log('  спектр T: кути повороту, гістограма 0..180° по 30°: {}'.format(h))
        log('  слід T = {:.3f}  (для випадкового повороту ≈0, для −I = −256)'.format(float(torch.diagonal(Qd).sum())))
        sym=float((Qd-Qd.t()).abs().mean()/Qd.abs().mean())
        log('  несиметричність ‖T−Tᵀ‖/‖T‖ = {:.3f}  (0 = дзеркало, 2 = чистий поворот)'.format(sym))
        # ---- дешеві кандидати З ВАГ ----
        W0=W[0].double()
        ee,V0=torch.linalg.eigh(W0@W0.t())
        for nm,Tc in (('власні вектори W₀W₀ᵀ, знак за медіаною',
                        (V0*torch.where(ee>ee.median(),1.0,-1.0)).matmul(V0.t())),
                      ('дзеркало вздовж головної моди W₀',
                        torch.eye(D,dtype=torch.float64,device=dev)-2*torch.outer(V0[:,-1],V0[:,-1]))):
            log('  дешевий кандидат [{:s}]: ρ = {:+.4f}'.format(nm,rho_of(W,Tc.float(),[960001,960002])))
        # ---- наскільки T близька до знайденої з ваг? ----
        log('  збіг знайденої T з найкращим дешевим: ⟨T, T_cheap⟩/d = {:.3f}'
            .format(float((Qd.double()*((V0*torch.where(ee>ee.median(),1.0,-1.0)).matmul(V0.t()))).sum()/D)))
log('done.')
