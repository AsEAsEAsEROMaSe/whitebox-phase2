# С111-д: ДІАГНОСТИКА ПРОВАЛУ. Стенд обіцяв +6.6%, повний оцінювач дав ×0.0001.
# Дві підозри, і вони РОЗДІЛЬНІ:
#   (1) КРИТЕРІЙ: стенд ранжував за ЧАСТОТОЮ frq, сабміт — за `mx` (максимум
#       передактивації по 512 пілотних рядках), тобто за МАСШТАБОМ нейрона
#   (2) ГЛОБАЛЬНО vs ПОШАРОВО: стенд ріже 22% з усіх 7936, сабміт — рівно 22%
#       на КОЖНОМУ шарі (а за С109 «відповідь народжується РАНО, шари 0-7 = 108%»)
# Міряємо всі чотири комбінації в ОДНІЙ таблиці.
import numpy as np
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(4242); NET=6; NMC=32768; PILOT=256; F=0.22
def price(alive):
    tot=0.0; prev=256
    for l in range(32):
        out=int(alive[l].sum()) if l<31 else 256
        tot+=prev*out; prev=out
    return 2.0*tot
def fwd_mean(z,Wn,alive):
    h=np.maximum(z@Wn[0],0.0)
    if alive is not None: h=h*alive[0]
    for l in range(1,32):
        h=np.maximum(h@Wn[l],0.0)
        if alive is not None and l<31: h=h*alive[l]
    return h.mean(0)
rows={}
for n in range(NET):
    Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    zp=rng.standard_normal((PILOT,256)); xp=np.concatenate([zp,-zp])
    h=np.maximum(xp@Wn[0],0.0)
    frq=np.zeros((31,256)); mx=np.zeros((31,256))
    frq[0]=(xp@Wn[0]>0).mean(0); mx[0]=(xp@Wn[0]).max(0)
    for l in range(1,31):
        p=h@Wn[l]; frq[l]=(p>0).mean(0); mx[l]=p.max(0); h=np.maximum(p,0.0)
    z=rng.standard_normal((NMC,256)); full=fwd_mean(z,Wn,None)
    p_full=price([np.ones(256,bool)]*31)
    def run(name,alive):
        pm=fwd_mean(z,Wn,[alive[l].astype(np.float64) for l in range(31)])
        rows.setdefault(name,[]).append((((pm-full)**2).mean(),
                                         price([alive[l] for l in range(31)])/p_full))
    for cname,val in [('frq',frq),('mx',mx)]:
        # ГЛОБАЛЬНО
        al=np.ones((31,256),bool); al.reshape(-1)[np.argsort(val,axis=None)[:int(F*31*256)]]=False
        run(f'{cname} ГЛОБАЛЬНО @0.22',al)
        # ПОШАРОВО (як у сабміті)
        al=np.ones((31,256),bool); k=int(F*256)
        for l in range(31): al[l][np.argsort(val[l])[:k]]=False
        run(f'{cname} ПОШАРОВО @0.22',al)
    # 🔒 база сабміту
    al=(frq>0.0); run('🔒БАЗА frq==0',al)
    print(f'мережа {n}',flush=True)
print(f"\n{'варіант':<26}{'зсув²':>13}{'ціна/повна':>12}{'× проти бази':>14}")
b=np.array(rows['🔒БАЗА frq==0'])[:,0].mean()
for k in ['🔒БАЗА frq==0','frq ГЛОБАЛЬНО @0.22','frq ПОШАРОВО @0.22','mx ГЛОБАЛЬНО @0.22','mx ПОШАРОВО @0.22']:
    v=np.array(rows[k]); print(f'{k:<26}{v[:,0].mean():>13.4e}{v[:,1].mean():>12.4f}{v[:,0].mean()/b:>14.2f}')
