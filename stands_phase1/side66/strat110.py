# С110: чи вказують ваги на ОСМИСЛЕНІ ЦЕНТРОЇДИ для реальних семплів (ідея ЮЗЕРА).
# Страта = argmax_j (z . w_j_норм), тобто чия комірка Вороного. Ціна = 1 матмул з 32 (вже рахуємо).
# Редукція дисперсії при пропорційній стратифікації = 1/(1-R2).
# ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ: R2 = 0.5..3%, контроль такий самий, x1.005..1.03. Шанс R2>10%: ~12%.
import numpy as np, time
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(555); NET=4; NS=65536
def fwd(h,Wn):
    for l in range(32): h=np.maximum(h@Wn[l],0.0)
    return h
def r2(lab,O,K):
    o=O.mean(0); tot=((O-o)**2).sum(); btw=0.0
    for j in range(K):
        m=lab==j; n=int(m.sum())
        if n: btw+=n*((O[m].mean(0)-o)**2).sum()
    return btw/tot
res={}
for n in range(NET):
    t0=time.time(); Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    U=Wn[0]/np.linalg.norm(Wn[0],axis=0,keepdims=True)         # центроїди = напрямки нейронів шару 0
    R=rng.standard_normal((256,256)); R/=np.linalg.norm(R,axis=0,keepdims=True)
    O=[];lw=[];lr=[]
    for i in range(0,NS,8192):
        z=rng.standard_normal((8192,256))
        O.append(fwd(z.copy(),Wn)); lw.append(np.argmax(z@U,axis=1)); lr.append(np.argmax(z@R,axis=1))
    O=np.vstack(O); lw=np.concatenate(lw); lr=np.concatenate(lr)
    lrand=rng.integers(0,256,size=NS)
    res.setdefault('ваги (центроїди ЮЗЕРА)',[]).append(r2(lw,O,256))
    res.setdefault('🔒випадкові напрямки',[]).append(r2(lr,O,256))
    res.setdefault('🔒випадкове призначення',[]).append(r2(lrand,O,256))
    # чи рівномірні комірки
    cnt=np.bincount(lw,minlength=256)
    res.setdefault('_занятість',[]).append((cnt.min(),cnt.max(),cnt.std()/cnt.mean()))
    print(f'мережа {n} {time.time()-t0:.0f}s',flush=True)
print('\n=== R2: ЧАСТКА ДИСПЕРСІЇ ВИХОДУ, ЯКУ ПОЯСНЮЄ НОМЕР КОМІРКИ ===')
for k in ['ваги (центроїди ЮЗЕРА)','🔒випадкові напрямки','🔒випадкове призначення']:
    a=np.mean(res[k]); print(f'  {k:<28} R2 = {a*100:6.3f}%   редукція x{1/(1-a):.4f}')
c=np.array(res['_занятість']).mean(0)
print(f'\n  занятість комірок: min {c[0]:.0f} max {c[1]:.0f} (рівномірно={NS/256:.0f}) розкид {c[2]*100:.0f}%')
print(f'  🔒підлога R2 від скінченності вибірки: ~255/{NS} = {255/NS*100:.3f}%')
