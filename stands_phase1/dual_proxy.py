"""СЕМПЛИ Й ВАГИ РІВНОПРАВНІ ⟹ ПРОКСІ МУСИТЬ ДИВИТИСЬ НА ЇХНЮ ВЗАЄМОДІЮ (ідея юзера, С47)

ДВОЇСТІСТЬ: h = x·W, у сумі Σ x_i W_ij семпл і вага рівноправні. Поворот вибірки X→XO
тотожний повороту ваг W₀→OᵀW₀ — приз 3.25× у поворотах вибірки ТОЙ САМИЙ, що у поворотах ваг.
АСИМЕТРІЯ: ваги ми ЗНАЄМО ТОЧНО, семпли ОБИРАЄМО. ⟹ проксі має бути функцією ВЗАЄМОДІЇ.

ЧОМУ ВЧОРАШНІЙ ПРОКСІ БУВ КАЛІКОЮ: арксинус дивився лише на ЗНАКИ передактивацій
(затвор відкритий/закритий) і викидав ЗНАЧЕННЯ. А далі мережею тече саме значення.

НОВИЙ ПРОКСІ C: беремо те, що РЕАЛЬНО виходить із 1-го шару — relu(h). Для гаусового входу
   E[relu(h_j)] = σ_j/√(2π)  ТОЧНО (h_j зі середнім 0), σ_j² = (W₀ᵀW₀)_jj. Безкоштовно.
   На антитетичних парах середнє = ½|h| — те саме.
   proxy_C = ‖ mean_емп(½|h|) − σ/√(2π) ‖   (вектор по 256 нейронах)
Це ВЖЕ НЕ сигнал про затвори, а САМА ВЕЛИЧИНА, що тече далі.

ПРОКСІ D (контроль, глибший): те саме, але для 2-го шару з ЕМПІРИЧНОЮ σ — цілі неточні,
   має працювати ГІРШЕ, якщо справа саме в точності цілі.
ПРОКСІ A (старий, контроль): розбіжність половин — має лишитись сліпим.
"""
import torch, time, numpy as np, math
LOG='/mnt/win_d/llm/White-Box/dual_proxy.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; M=200
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(W,x):
    a=x
    for li in range(L): a=torch.relu(a@W[li])
    return a
def zca(x):
    C=(x.t()@x)/x.shape[0]
    e,V=torch.linalg.eigh(C.double().cpu())
    return (V@torch.diag(e.clamp(min=1e-14).rsqrt())@V.t()).to(dev).float()
def rand_orth(seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    A=torch.randn(D,D,device=dev,generator=g)
    Q,R=torch.linalg.qr(A.double())
    return (Q*torch.sign(R.diag())[None,:]).float()

def run(net,NP=100_000):
    W=load(net)
    S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(40):
        g=torch.Generator(device=dev).manual_seed(6100+c)
        z=torch.randn(100_000,D,device=dev,generator=g)
        xx=torch.cat([z,-z],0); S+=fwd(W,xx).double().sum(0); n+=xx.shape[0]
    TH=S/n
    log('  істина (8М): ‖E‖={:.4f}'.format(float(TH.norm())))
    # ТОЧНА ціль 1-го шару
    sig=(W[0].double().t()@W[0].double()).diag().sqrt()
    tgt1=sig/math.sqrt(2*math.pi)
    log('  точна ціль 1-го шару: ‖E[a¹]‖={:.4f} (формула σ/√(2π), безкоштовно)'.format(float(tgt1.norm())))
    g=torch.Generator(device=dev).manual_seed(31337)
    z=torch.randn(NP,D,device=dev,generator=g)
    X=torch.cat([z,-z],0); X=X@zca(X)
    err=[];pA=[];pC=[];pD=[]
    for m in range(M):
        Xo=X@rand_orth(20000+m)
        h1=Xo@W[0]
        a1=torch.relu(h1)
        m1=((a1[:NP]+a1[NP:])*0.5).double().mean(0)
        pC.append(float((m1-tgt1).norm()))                      # ПРОКСІ C — точна ціль
        h2=a1@W[1]
        a2=torch.relu(h2)
        m2=((a2[:NP]+a2[NP:])*0.5).double().mean(0)
        s2=h2.double().std(0)                                    # ЕМПІРИЧНА σ (ціль неточна)
        pD.append(float((m2-s2/math.sqrt(2*math.pi)).norm()))    # ПРОКСІ D — контроль
        a=a1
        for li in range(1,L): a=torch.relu(a@W[li])
        p=((a[:NP]+a[NP:])*0.5).double()
        err.append(float((p.mean(0)-TH).norm()))
        hh=NP//2; pA.append(float((p[:hh].mean(0)-p[hh:].mean(0)).norm()))
    e=np.array(err); se=1/ (M-3)**0.5
    log('  помилка: сер={:.4e} мін={:.4e} макс={:.4e} (приз {:.2f}× по MSE)'
        .format(e.mean(),e.min(),e.max(),(e.mean()/e.min())**2))
    log('  похибка вимірювання кореляції ±{:.3f}'.format(se))
    for nm,p in (('A розбіжність половин (контроль)',pA),
                 ('C 1-й шар, ТОЧНА ціль σ/√(2π)',pC),
                 ('D 2-й шар, емпірична ціль',pD)):
        c=float(np.corrcoef(np.array(p),e)[0,1])
        log('   {:36s} корел={:+.4f}   {}'.format(nm,c,'*** БАЧИТЬ ***' if abs(c)>3*se else 'сліпий'))

for net in (0,1):
    log('==== НЕТ {} ===='.format(net)); run(net)
log('done.')
