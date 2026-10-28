"""ІДЕЯ ЮЗЕРА: подати вихід мережі назад на вхід (як зациклювання LLM).

ЯК СЕМПЛ — НЕЗАКОННО: a^L ≥ 0, не гаусів ⟹ прямий зсув.
ЯК ДИНАМІЧНА СИСТЕМА — законно й БЕЗКОШТОВНО: мережа без bias ОДНОРІДНА, T(cx)=c·T(x),
тож ітерація x̂→T(x̂)/|T(x̂)| — степеневий метод на сфері. Потребує лише ВАГИ, нуль семплів
⟹ [[finding-anchor-cost-identity]] (податок якоря) на неї НЕ ДІЄ.

СТАВКА: якщо E[a^L] колінеарний атрактору v*, оцінюємо ОДНЕ число (довжину) замість 256,
і шум із 255 ортогональних осей викидається ⟹ до 256×.
  MSE_проєкції = (шум по k осях)/256 + |TH|²(1−cos²)/256   ← другий доданок = ЗСУВ, він і вирішує

МІРЯЄМО:
  1) чи сходиться ітерація і чи в ТУ САМУ точку з різних стартів
  2) cos(v*, TH) — колінеарність із істинною відповіддю
  3) підпростір із k ітератів (крилівський) — скільки TH у ньому лежить
  4) КОНТРОЛЬ: випадковий підпростір тієї ж вимірності
  5) підсумок: MSE-підлога схеми проти теперішніх 2.5e-6
"""
import torch, time, math, numpy as np
LOG='/mnt/win_d/llm/White-Box/loopback.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(W,x):
    a=x
    for i in range(L): a=torch.relu(a@W[i])
    return a

for net in (0,1):
    W=load(net); log('='*84); log('МЕРЕЖА %d'%net)
    S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(30):
        g=torch.Generator(device=dev).manual_seed(990000+c)
        z=torch.randn(60000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        S+=fwd(W,xx).double().sum(0); n+=xx.shape[0]
    TH=S/n; nTH=float(TH.norm())
    log('  |E[a^L]| = {:.4f};  теперішній MSE = 2.5e-6  ⟹  відн. похибка {:.3f}%'
        .format(nTH,100*math.sqrt(2.5e-6*256)/nTH))

    # --- 1) ітерація з 8 різних стартів ---
    g=torch.Generator(device=dev).manual_seed(4242)
    X=torch.randn(8,D,device=dev,generator=g)
    X=X/X.norm(dim=1,keepdim=True)
    traj=[]
    prev=None
    log('  ітер |Δ напрямку|   cos(v_iter, TH)   розкид між стартами')
    for it in range(500):
        Y=fwd(W,X); nz=Y.norm(dim=1,keepdim=True).clamp(min=1e-30); X=Y/nz
        traj.append(X[0].clone())
        d=float((X-prev).norm(dim=1).mean()) if prev is not None else float('nan')
        prev=X.clone()
        cosv=(X@TH.float()/TH.float().norm())
        G=X@X.t(); off=float((G-torch.eye(8,device=dev)).abs().max())
        if it in (0,1,2,3,5,9,19,39,79,159,249,349,449,499):
            log('  {:4d} {:12.3e} {:18.4f} {:18.4f}'.format(it,d,float(cosv.mean()),1-off))
    v=X[0]
    log('  ⟹ атрактор: усі 8 стартів зійшлись у ту саму точку? розбіжність {:.2e}'
        .format(float((X-X[0:1]).norm(dim=1).max())))

    # --- 2,3) підпростори ---
    def floor_mse(B,tag):
        Q,_=torch.linalg.qr(B)
        proj=Q@(Q.t()@TH.float())
        bias=float(((TH.float()-proj)**2).mean())
        k=Q.shape[1]
        noise=2.5e-6*k/256.0        # шум лишається лише в k осях
        log('  {:34s} k={:3d}  зсув {:9.3e}  шум {:9.3e}  MSE {:9.3e}  ({:5.2f}× до 2.5e-6)'
            .format(tag,k,bias,noise,bias+noise,2.5e-6/(bias+noise)))
    for k in (1,2,4,8,16):
        B=torch.stack(traj[-k:],1) if k>1 else v.unsqueeze(1)
        floor_mse(B,'A) останні {:d} ітерати'.format(k))
    for k in (1,4,16):
        B=torch.stack(traj[:k],1)
        floor_mse(B,'B) ПЕРШІ {:d} ітерати (перехід)'.format(k))
    for k in (1,4,16):
        gg=torch.Generator(device=dev).manual_seed(31+k)
        floor_mse(torch.randn(D,k,device=dev,generator=gg),'C) ВИПАДКОВІ — КОНТРОЛЬ')
log('done.')
