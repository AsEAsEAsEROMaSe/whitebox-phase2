# 🔒 ПЛОМБА НА МЕХАНІЗМ: чи справді вибілювання ЗАНУЛЯЄ контрольну змінну
# g = mean(relu(p0)) − ‖w‖/√(2π)? Якщо так, ×1.0096 пояснено, і клас закрито
# не «емпірично», а причиною. ПЕРЕДБАЧЕННЯ: ‖g‖² падає у 5..30×.
import numpy as np
W=np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy',mmap_mode='r')
rng=np.random.default_rng(99); NET=8; NPAIR=3072; REP=64
INV=0.3989422804014327
raw=[];whi=[];rawA=[];whiA=[]
for n in range(NET):
    W0=np.array(W[n,0],dtype=np.float64); sg0=np.linalg.norm(W0,axis=0)
    E=sg0*INV
    a=[];b=[];c=[];d=[]
    for r in range(REP):
        z=rng.standard_normal((NPAIR,256))
        # гола МК тієї ж ефективної кількості семплів
        zz=rng.standard_normal((2*NPAIR,256))
        a.append((((np.maximum(zz@W0,0.0)).mean(0)-E)**2).sum())
        # лише антитетика
        y0=z@W0; x0=np.concatenate([np.maximum(y0,0),np.maximum(-y0,0)])
        c.append(((x0.mean(0)-E)**2).sum())
        # вибілювання + антитетика (як у сабміті)
        cov=(z.T@z)/NPAIR; wh=np.linalg.inv(np.linalg.cholesky(cov)).T
        y=z@(wh@W0); x=np.concatenate([np.maximum(y,0),np.maximum(-y,0)])
        b.append(((x.mean(0)-E)**2).sum())
        # 🔒 контроль: та сама формула, але ЛІНІЙНА ознака замість relu
        #    (для неї вибілювання мусить давати РІВНО 0 — пломба приладу)
        d.append((((z@(wh@W0)).mean(0))**2).sum())
    raw.append(np.mean(a));whi.append(np.mean(b));rawA.append(np.mean(c));whiA.append(np.mean(d))
r=np.mean(raw);w_=np.mean(whi);ra=np.mean(rawA);lin=np.mean(whiA)
print(f"{'варіант':<34}{'‖g‖²':>14}{'× проти голої':>15}")
print(f"{'гола МК':<34}{r:>14.4e}{1.0:>15.3f}")
print(f"{'+ антитетика':<34}{ra:>14.4e}{r/ra:>15.3f}")
print(f"{'+ антитетика + ВИБІЛЮВАННЯ':<34}{w_:>14.4e}{r/w_:>15.3f}")
print(f"\n🔒 пломба приладу: ЛІНІЙНА ознака після вибілювання = {lin:.3e} (мусить бути ~0)")
