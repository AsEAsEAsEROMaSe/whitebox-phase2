"""НАЙСИЛЬНІША ВЕРСІЯ ідеї «сусіди в орбіті»: шукаємо T ОПТИМІЗАЦІЄЮ по всій ортогональній групі.

orbits.py перевірив 10 вгаданих перетворень — усі ρ>0 (мінімум +0.058 = фактично незалежні).
Але група має d(d−1)/2 = 32640 параметрів. 10 точок ≠ відповідь (та сама помилка «дивитись
у шпаринку», що вже тричі підвела сьогодні).

ТУТ: градієнтний пошук T, що МІНІМІЗУЄ ρ = corr(y_even(x), y_even(xT)).
  • ортогональність тримаємо через QR-репроєкцію щокроку
  • оцінка ½(y(x)+y(xT)) НЕЗМІЩЕНА для будь-якого фіксованого ортогонального T
    (бо xT теж стандартний гаус) ⟹ T можна вільно підбирати, це лише вибір системи координат
  • ⚠️ T підганяється на ОДНИХ семплах, ρ міряється на ІНШИХ (інакше переобладнання)

ТЕОРІЯ КАЖЕ, ЩО ВІД'ЄМНЕ ρ МОЖЛИВЕ: для квадратичної частини xᵀAx маємо
corr = ⟨A,TATᵀ⟩/‖A‖²; якщо T міняє місцями додатний і від'ємний власні підпростори A,
то TATᵀ=−A і ρ=−1. Питання в тому, чи це досяжно ОДНИМ T одразу для 256 координат
і для 4-го порядку, який (виміряно) РОЗМАЗАНИЙ.
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/orbit_opt.log'; T0=time.time()
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
    """зважена дисперсією кореляція між y(x) і y(xT) на заданих сідax"""
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

for net in (0,):
    W=load(net); log('='*80); log('МЕРЕЖА %d'%net)
    g0=torch.Generator(device=dev).manual_seed(2024)
    M=torch.randn(D,D,device=dev,generator=g0)*0.05
    M.requires_grad_(True)
    opt=torch.optim.Adam([M],lr=0.02)
    base,_=torch.linalg.qr(torch.randn(D,D,device=dev,generator=g0))
    CH=4000
    log('  старт: ρ випадкового повороту = {:+.4f}'.format(rho_of(W,base,[900001,900002])))
    log('')
    log('  {:>5s} {:>12s} {:>14s}'.format('крок','ρ (навчання)','ρ (свіжі)'))
    for it in range(121):
        Q,R=torch.linalg.qr(base+M)                       # ортогональна, диференційовна
        Q=Q*torch.sign(torch.diagonal(R)).unsqueeze(0)
        g=torch.Generator(device=dev).manual_seed(700000+it)
        z=torch.randn(CH,D,device=dev,generator=g)
        a=even(W,z); b=even(W,z@Q)
        ac=a-a.mean(0,keepdim=True); bc=b-b.mean(0,keepdim=True)
        cov=(ac*bc).sum(); va=(ac*ac).sum(); vb=(bc*bc).sum()
        r=cov/torch.sqrt(va*vb+1e-30)
        opt.zero_grad(); r.backward(); opt.step()
        if it%20==0:
            with torch.no_grad():
                Qd,Rd=torch.linalg.qr(base+M)
                Qd=Qd*torch.sign(torch.diagonal(Rd)).unsqueeze(0)
                fresh=rho_of(W,Qd.detach(),[950001,950002,950003])
            log('  {:5d} {:+12.4f} {:+14.4f}'.format(it,float(r),fresh))
    with torch.no_grad():
        Qd,Rd=torch.linalg.qr(base+M); Qd=Qd*torch.sign(torch.diagonal(Rd)).unsqueeze(0)
        fin=rho_of(W,Qd.detach(),[960001,960002,960003,960004])
        orth=float((Qd.t()@Qd-torch.eye(D,device=dev)).abs().max())
    log('')
    log('  ПІДСУМОК: найкраще ρ поза вибіркою = {:+.4f}  ⟹ виграш {:.3f}×'.format(fin,1.0/(1.0+fin)))
    log('  перевірка ортогональності ‖QᵀQ−I‖∞ = {:.2e}'.format(orth))
    log('  (виграш >1 лише при ρ<0; ρ≈0 означає «не краще за незалежний семпл»)')
log('done.')
