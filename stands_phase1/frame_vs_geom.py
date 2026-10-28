"""ВАГИ = ЛИШЕ ВІДСТАНІ, ЧИ ЩЕ Й ОРІЄНТАЦІЯ? (ідея юзера С47: «ваги просто розводять дані»)

ТЕОРІЯ: W -> O@W (O ортогональна) зберігає ГЕОМЕТРІЮ потоків ТОЧНО: (OW)^T(OW)=W^T W.
Змінюється лише орієнтація відносно ОСЕЙ КООРДИНАТ — а ReLU прив'язаний саме до осей.
  * шар 0: вхід ізотропний, xO ~ x  =>  E[a^L] НЕ ЗМІНЮЄТЬСЯ (ТЕОРЕМА = вбудований контроль)
  * шар l>=1: вхід уже пройшов ReLU, НЕ ізотропний => орієнтація може важити

МІРЯЄМО для кожного шару l: розкид ||E_rot - E_base|| по кількох випадкових O,
нормований на РОЗКИД МІЖ РІЗНИМИ МЕРЕЖАМИ (масштаб «усієї різниці»).
ЧИТАННЯ: мала частка => шар описується своєю матрицею Грама (стиснення);
         велика     => важить орієнтація відносно осей, самих відстаней НЕ досить.
Спільні випадкові числа (той самий сід) => різниця не забруднена шумом MC.
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/frame_vs_geom.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def E(W,N=2_000_000,CH=100_000,seed=8800):
    s=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(N//CH):
        gg=torch.Generator(device=dev).manual_seed(seed+c)         # СПІЛЬНІ вип. числа
        z=torch.randn(CH//2,D,device=dev,generator=gg)
        a=torch.cat([z,-z],0)
        for li in range(L): a=torch.relu(a@W[li])
        s+=a.double().sum(0); n+=a.shape[0]
    return s/n

def rand_orth(seed):
    gg=torch.Generator(device=dev).manual_seed(seed)
    A=torch.randn(D,D,device=dev,generator=gg)
    Q,R=torch.linalg.qr(A.double())
    return (Q*torch.sign(R.diag())[None,:]).float()

net=0
W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
Ebase=E(W); nb=float(Ebase.norm())
log('база: ||E[a^L]|| = {:.4f}'.format(nb))

# масштаб: розкид МІЖ мережами
others=[]
for m in (1,2,3):
    Wm=[torch.tensor(dat['w_%d_%d'%(m,i)].astype(np.float32),device=dev) for i in range(L)]
    others.append(float((E(Wm)-Ebase).norm())/nb)
scale=float(np.mean(others))
log('масштаб «уся різниця між мережами»: {:.4f} (нети 1,2,3: {})'.format(
    scale,' '.join('{:.3f}'.format(o) for o in others)))
log('')
log('  шар    відн.зсув від повороту (3 незалежні O)      частка від «між мережами»')
for l in (0,1,2,3,4,8,16,24,31):
    ds=[]
    for r in range(3):
        O=rand_orth(9000+37*l+r)
        Wr=list(W); Wr[l]=O@W[l]
        ds.append(float((E(Wr)-Ebase).norm())/nb)
    mu=float(np.mean(ds))
    tag=''
    if l==0: tag='   <<< ТЕОРЕМА: має бути ~0 (контроль)'
    log('   {:2d}     {}      {:.3f}{}'.format(l,' '.join('{:.4f}'.format(d) for d in ds),mu/scale,tag))
log('done.')
