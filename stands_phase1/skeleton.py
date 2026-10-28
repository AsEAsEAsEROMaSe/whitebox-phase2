"""СКЕЛЕТ: викинути ваги, лишити ReLU (ідея юзера С47).

W = U S V^T:  S = РОЗТЯГ («на скільки дані стають одні від одних»), U V^T = ПОВОРОТ.
«Викинути ваги» = зрівняти всі розтяги: W_skel = sqrt(2) * U V^T (ортогональна, той самий масштаб).
Лишається чистий скелет: поворот + зріз по осях. У випадкової W розтяги розкидані широко
(Марченко-Пастур, від 0 до ~2*sqrt(2)) => викидаємо БАГАТО.

ШЛЯХ: S_t = (1-t)*S + t*sqrt(2)  для t=0(оригінал)..1(скелет).

МІРЯЄМО:
 1) ||E_t - E_real|| / масштаб «зовсім інша мережа»  — скільки тотожності мережі в розтягах
 2) посемплова КОРЕЛЯЦІЯ a^L_t з a^L_real (спільні вип. числа) — чи вийде контрольна змінна
 3) дисперсія MC оцінювача для t — чи скелет ТИХІШИЙ
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/skeleton.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
S2=2.0**0.5

def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]

def morph(W,t):
    """ВАЖЛИВО: після зрівнювання розтягів ПРИМУСОВО відновлюємо критичність (E[s^2]=2),
    інакше мережа стає підкритичною і падіння дисперсії — артефакт затухання, а не скелета."""
    out=[]
    for w in W:
        U,S,Vh=torch.linalg.svd(w.double())
        St=(1-t)*S+t*S2
        St=St*( (2.0*D/ (St*St).sum()) )**0.5          # E[s^2]=2 точно
        out.append((U@torch.diag(St)@Vh).float())
    return out

def stats(W,Wref=None,N=2_000_000,CH=100_000,seed=8800):
    """повертає E[a^L], дисперсію на пару, і (якщо є Wref) кореляцію з ним"""
    s=torch.zeros(D,dtype=torch.float64,device=dev); ss=0.0; n=0
    sx=torch.zeros(D,dtype=torch.float64,device=dev); sxy=0.0; syy=0.0; sy=torch.zeros(D,dtype=torch.float64,device=dev)
    for c in range(N//CH):
        gg=torch.Generator(device=dev).manual_seed(seed+c)
        z=torch.randn(CH//2,D,device=dev,generator=gg)
        x=torch.cat([z,-z],0)
        a=x
        for li in range(L): a=torch.relu(a@W[li])
        p=((a[:CH//2]+a[CH//2:])*0.5).double()          # парна частина пари
        s+=p.sum(0); ss+=float((p*p).sum()); n+=p.shape[0]
        if Wref is not None:
            b=x
            for li in range(L): b=torch.relu(b@Wref[li])
            q=((b[:CH//2]+b[CH//2:])*0.5).double()
            sx+=p.sum(0); sy+=q.sum(0); sxy+=float((p*q).sum()); syy+=float((q*q).sum())
    E=s/n; var=ss/n-float((E*E).sum())
    corr=None
    if Wref is not None:
        Ex=sx/n; Ey=sy/n
        cxy=sxy/n-float((Ex*Ey).sum()); vx=var; vy=syy/n-float((Ey*Ey).sum())
        corr=cxy/(vx*vy)**0.5
    return E,var,corr

W=load(0)
Ereal,varreal,_=stats(W)
nb=float(Ereal.norm())
log('оригінал: ||E||={:.4f}  дисперсія на пару={:.4e}'.format(nb,varreal))
sc=[]
for m in (1,2,3):
    Em,_,_=stats(load(m)); sc.append(float((Em-Ereal).norm())/nb)
scale=float(np.mean(sc))
log('масштаб «зовсім інша мережа» = {:.4f}'.format(scale))
sv=torch.linalg.svdvals(W[0].double())
log('розтяги шару 0: min={:.3f} медіана={:.3f} max={:.3f}  (скелет ставить усі = {:.3f})'
    .format(float(sv.min()),float(sv.median()),float(sv.max()),S2))
log('')
log('   t      ||E_t-E_real||/масштаб    корел. з оригіналом    дисперсія/оригінал')
for t in (0.25,0.5,0.75,1.0):
    Wt=morph(W,t)
    Et,vt,ct=stats(Wt,Wref=W)
    log('  {:.2f}        {:.3f}                  {:+.4f}              {:.3f}'
        .format(t,float((Et-Ereal).norm())/nb/scale,ct,vt/varreal))
log('done.')
