# С111: чи є ДЕШЕВА ознака з ПЕРШОГО шару, що передбачає вихід краще за наш
# λ-блок. Це наша версія відкритого питання поля O6/O8 (мережезалежний вибір
# фреймів): bulzan мав оракул 95.64%, законний проксі cos = 0.018.
# Усі ознаки рахуються з p0 = z@W0 (1 матвектор = 1/32 проходу).
# ПЕРЕДБАЧЕННЯ: скалярні ознаки дадуть R2 < 5%; покомпонентний relu(p0)
# (=λ-блок) дасть ~11%; НІЧОГО не поб'є λ-блок. Шанс на >20%: ~15%.
import numpy as np
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(4242); NET=6; NS=32768

def r2_multi(F, Y):
    """R² багатовимірної регресії Y (n×256) на F (n×k) з вільним членом."""
    F=np.column_stack([np.ones(len(F)),F])
    beta,_,_,_=np.linalg.lstsq(F,Y,rcond=None)
    res=Y-F@beta
    return 1.0-(res**2).sum()/((Y-Y.mean(0))**2).sum()

acc={}
for n in range(NET):
    Wn=[np.array(W[n,l],dtype=np.float64) for l in range(32)]
    P0=[];OUT=[]
    for i in range(0,NS,8192):
        z=rng.standard_normal((8192,256))
        z=z-z.mean(0)                                  # вибілювання, як у сабміті
        cov=z.T@z/len(z); z=z@np.linalg.inv(np.linalg.cholesky(cov)).T
        p0=z@Wn[0]; h=np.maximum(p0,0.0)
        for l in range(1,32): h=np.maximum(h@Wn[l],0.0)
        P0.append(p0); OUT.append(h)
    p0=np.vstack(P0); Y=np.vstack(OUT)
    rp=np.maximum(p0,0.0)
    feats={
      'норма ‖p0‖':            np.linalg.norm(p0,axis=1)[:,None],
      'к-сть відкритих':       (p0>0).sum(1)[:,None].astype(float),
      'Σ relu(p0)  (скаляр)':  rp.sum(1)[:,None],
      'max p0':                p0.max(1)[:,None],
      'Σ p0²':                 (p0*p0).sum(1)[:,None],
      'усі 5 скалярів разом':  np.column_stack([np.linalg.norm(p0,axis=1),(p0>0).sum(1),
                                                rp.sum(1),p0.max(1),(p0*p0).sum(1)]),
      'топ-8 власних W0ᵀW0':   p0@np.linalg.eigh(Wn[0].T@Wn[0])[1][:,-8:],
      'λ-БЛОК: relu(p0) ×256': rp,
      '🔒p0 покомпонентно ×256': p0,
    }
    for k,F in feats.items():
        acc.setdefault(k,[]).append(r2_multi(F,Y))
    print(f'мережа {n} готова',flush=True)
print()
print(f"{'ознака (усе з 1 матвектора z@W0)':<28}{'R² проти виходу':>18}{'sd':>9}")
for k,v in acc.items():
    print(f'{k:<28}{np.mean(v)*100:>17.2f}%{np.std(v)*100:>8.2f}')
