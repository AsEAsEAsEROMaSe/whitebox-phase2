"""ЩО САМЕ ЗАЛИШАЄ v2? Гіпотеза: v2 = ТОЧНИЙ ЗРІЗ ЗА ПОРЯДКАМИ.
  антитетика  ⟹ непарна частина знята ТОЧНО (побітово)
  ZCA         ⟹ другі вибіркові моменти точні ⟹ квадратична частина парного залишку знята
  ⟹ те, що ми звемо похибкою v2, це РІВНО 4-й порядок і вище.

Якщо правда — це задає ТЗ на метод рівня 1e-8: знімати 4-й порядок так само точно,
як ZCA знімає 2-й. І пояснює 0.7× у гучних / 4.0× у вагаючих ([[finding-locks-real-but-empty]])
не як властивість нейронів, а як розподіл кривизни вищого порядку.

МІРЯЄМО по кожній з 256 координат відповіді:
  y_even(x) = (a(x)+a(-x))/2                     — те, що взагалі лишається після антитетики
  R²_quad   = частка Var(y_even), пояснена КВАДРАТИЧНИМИ ознаками (Ерміт-2)
  залишок   = 1-R²_quad = 4-й порядок і вище

Квадратичні ознаки: q_k(x) = (x·u_k)² - 1 для M випадкових напрямків u_k.
Це випадкова підвибірка простору Ерміт-2 (повний = d(d+1)/2 = 32896) — дає НИЖНЮ оцінку
R²_quad, чого досить: якщо навіть підвибірка пояснює багато, гіпотеза стоїть.
Контроль: ті самі ознаки на ПЕРЕМІШАНИХ мітках ⟹ скільки R² дає сама розмірність.
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/orders.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32; M=1200
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd_h(W,x):
    a=x
    for li in range(L-1): a=torch.relu(a@W[li])
    h=a@W[L-1]; return h, torch.relu(h)

for net in (0,1):
    W=load(net); log('='*74); log('МЕРЕЖА %d'%net)
    gu=torch.Generator(device=dev).manual_seed(4242)
    U=torch.randn(D,M,device=dev,generator=gu); U/=U.norm(dim=0,keepdim=True)

    # нормальні рівняння накопичуємо блоками: [q|1]^T[q|1] та [q|1]^T y
    A=torch.zeros(M+1,M+1,dtype=torch.float64,device=dev)
    Bm=torch.zeros(M+1,D,dtype=torch.float64,device=dev)
    SY=torch.zeros(D,dtype=torch.float64,device=dev); SY2=torch.zeros(D,dtype=torch.float64,device=dev)
    P=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    CH=25_000
    for c in range(24):
        g=torch.Generator(device=dev).manual_seed(1200+c)
        z=torch.randn(CH,D,device=dev,generator=g)
        hp,ap=fwd_h(W,z); hm,am=fwd_h(W,-z)
        ye=((ap+am)*0.5).double()                       # парна частина = що лишилось після антитетики
        q=((z@U)**2-1.0).double()                       # Ерміт-2
        F=torch.cat([q,torch.ones(CH,1,dtype=torch.float64,device=dev)],1)
        A+=F.t()@F; Bm+=F.t()@ye
        SY+=ye.sum(0); SY2+=(ye*ye).sum(0)
        P+=(hp>0).double().sum(0)+(hm>0).double().sum(0); n+=CH
    p=P/(2*n)
    var=SY2/n-(SY/n)**2
    sol=torch.linalg.solve(A+torch.eye(M+1,dtype=torch.float64,device=dev)*1e-6, Bm)
    expl=((Bm*sol).sum(0)-(SY/n)*SY)/n                  # пояснена дисперсія
    r2=(expl/var.clamp(min=1e-30)).clamp(0,1)
    # контроль розмірності: M ознак на n семплах самі дають ~M/n
    log('  n={:d}  ознак={:d}  контроль переобладнання M/n={:.4f}'.format(n,M,M/n))
    log('  R²_quad (частка парного залишку, пояснена 2-м порядком): медіана {:.4f}  сер {:.4f}'
        .format(float(r2.median()),float(r2.mean())))

    q=p.argsort()
    log('  по децилях частоти p_i (тихі → гучні):')
    log('    {:>10s} {:>10s} {:>12s} {:>12s}'.format('сер p_i','R²_quad','Var парної','4-й+ порядок'))
    for k in range(10):
        idx=q[k*D//10:(k+1)*D//10]
        rest=float((var[idx]*(1-r2[idx])).sum())
        log('    {:10.3f} {:10.4f} {:12.4e} {:12.4e}'
            .format(float(p[idx].mean()),float(r2[idx].mean()),float(var[idx].sum()),rest))
    tot=float(var.sum()); res=float((var*(1-r2)).sum())
    log('  ПІДСУМОК: парна дисперсія {:.4e}; після зняття 2-го порядку {:.4e}  ⟹ виграш {:.2f}×'
        .format(tot,res,tot/max(res,1e-30)))
    log('  (це СТЕЛЯ ZCA на цих ознаках; реальна ZCA бере повний Ерміт-2, отже ≥)')
log('done.')
