"""ЧИ Є В ОПТИМІЗАТОРА ОЧІ? (ідея юзера С47: оптимізатор, що коригує семпли)

ЛОГІКА ПОБУДОВИ (як виводився v2): v2 не виправляє відповідь, а РУХАЄ СЕМПЛИ так, щоб вони
точно задовольняли відоме: антитетика -> 1-й момент точний, ZCA -> 2-й момент точний.
Доказ переваги руху над поправкою (С47): арксинус як ПОПРАВКА = 0%, а ZCA (та сама інформація,
але РУХОМ) = 22%.

ЗАЛИШЕНА СВОБОДА: ПОВОРОТИ. Гаус ізотропний => будь-який поворот вибірки лишає її правильною
вибіркою ТОЧНО; після ZCA поворот зберігає вибіленість. d(d-1)/2 = 32640 невикористаних
параметрів.

АЛЕ ОПТИМІЗАТОРУ ПОТРІБНІ ОЧІ: істини він не знає. Потрібен ПРОКСІ, що передбачає помилку.
ЦЕЙ СКРИПТ НЕ ОПТИМІЗУЄ. Він перевіряє, чи проксі взагалі БАЧИТЬ помилку:
  беремо ОДНУ вибірку, крутимо M випадковими поворотами, для кожного міряємо
    (1) СПРАВЖНЮ помилку ||est - truth||           (недоступна методу)
    (2) проксі A: розбіжність половин ||m_A - m_B|| (доступна)
    (3) проксі B: неузгодженість арксинуса          (доступна)
  і рахуємо кореляції (2)~(1) та (3)~(1).
ЧИТАННЯ: корел ~0 => оптимізатор СЛІПИЙ, будувати нема сенсу. Висока => будуємо.
Також: розкид помилки по поворотах = ЩО МОЖНА ВИГРАТИ у принципі (оракул best-of-M).
"""
import torch, time, numpy as np, math
LOG='/mnt/win_d/llm/White-Box/rot_eyes.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; M=200; MM=24
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]

def fwd(W,x):
    a=x
    for li in range(L): a=torch.relu(a@W[li])
    return a

def truth(W,N=8_000_000,CH=200_000,seed=6100):
    s=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(N//CH):
        gg=torch.Generator(device=dev).manual_seed(seed+c)
        z=torch.randn(CH//2,D,device=dev,generator=gg)
        x=torch.cat([z,-z],0); s+=fwd(W,x).double().sum(0); n+=x.shape[0]
    return s/n

def rand_orth(seed):
    gg=torch.Generator(device=dev).manual_seed(seed)
    A=torch.randn(D,D,device=dev,generator=gg)
    Q,R=torch.linalg.qr(A.double())
    return (Q*torch.sign(R.diag())[None,:]).float()

net=0; W=load(net); TH=truth(W)
log('істина: ||E||={:.4f}'.format(float(TH.norm())))
# арксинусні цілі для затворів 1-го шару
W0d=W[0].double(); Sig=W0d.t()@W0d; dgg=Sig.diag().rsqrt()
Aarc=torch.arcsin(((Sig*dgg[:,None])*dgg[None,:]).clamp(-1,1))/(2*math.pi)
ev,V=torch.linalg.eigh(Aarc.cpu()); lam=ev.flip(0)[:MM].to(dev); Vf=V.flip(1)[:,:MM].to(dev).float()
iu=torch.triu_indices(MM,MM,device=dev)
anc=torch.where(iu[0]==iu[1],lam[iu[0]],torch.zeros(iu.shape[1],dtype=lam.dtype,device=dev))

NP=100_000
gg=torch.Generator(device=dev).manual_seed(31337)
z=torch.randn(NP,D,device=dev,generator=gg)
X=torch.cat([z,-z],0)
C=(X.t()@X)/X.shape[0]
e_,Vz=torch.linalg.eigh(C.double().cpu())
X=X@(Vz@torch.diag(e_.clamp(min=1e-14).rsqrt())@Vz.t()).to(dev).float()   # ZCA
log('база вибілена: ||emp.cov - I||={:.2e}'.format(float(((X.t()@X)/X.shape[0]-torch.eye(D,device=dev)).norm())))

err=[];pA=[];pB=[]
for m in range(M):
    O=rand_orth(20000+m)
    Y=fwd(W,X@O)
    p=((Y[:NP]+Y[NP:])*0.5).double()
    est=p.mean(0)
    err.append(float((est-TH).norm()))
    h=NP//2
    pA.append(float((p[:h].mean(0)-p[h:].mean(0)).norm()))       # проксі A: розбіжність половин
    u=((X@O)@W[0]>0).float()-0.5
    q=(u[:NP]@Vf).double()
    f=(q[:,iu[0]]*q[:,iu[1]]-anc).mean(0)
    pB.append(float(f.norm()))                                    # проксі B: неузгодженість арксинуса
err=np.array(err);pA=np.array(pA);pB=np.array(pB)
log('')
log('помилка по {} поворотах: сер={:.4e}  мін={:.4e}  макс={:.4e}  (макс/мін={:.2f}×)'
    .format(M,err.mean(),err.min(),err.max(),err.max()/err.min()))
log('ОРАКУЛ best-of-{}: помилка {:.4e} => виграш {:.2f}× за нормою, {:.2f}× за MSE'
    .format(M,err.min(),err.mean()/err.min(),(err.mean()/err.min())**2))
log('')
log('  проксі A (розбіжність половин): корел зі справжньою помилкою = {:+.4f}'
    .format(float(np.corrcoef(pA,err)[0,1])))
log('  проксі B (неузгодж. арксинуса): корел зі справжньою помилкою = {:+.4f}'
    .format(float(np.corrcoef(pB,err)[0,1])))
log('')
log('ЧИТАННЯ: |корел| мала => оптимізатор СЛІПИЙ (крутити нема по чому, хоч виграш і існує).')
log('done.')
