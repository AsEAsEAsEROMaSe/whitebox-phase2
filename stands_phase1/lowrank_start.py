"""НИЗЬКОРАНГОВИЙ СИНТЕТИЧНИЙ СТАРТ — переоцінка [[finding-synthetic-deep-start]].

ПІДКАЗКА ОРГАНІЗАТОРІВ (town hall 10:46): «as the depth increases the effective rank of the
covariance matrix decreases ⟹ may be possible to recover the deficit».
ВИМІРЯНО (effrank.py): участь падає 165 → 20 → 6 → 2.0-3.4. Топ-8 мод = 85-90% дисперсії.

ЧОМУ ГІЛКА БУЛА ЗАКРИТА ПОМИЛКОВО: «K це 65536 чисел, не купується». Але при рангу 8 це
2048 чисел. І гірше: ми оцінювали K ПОВНОГО рангу з тисяч семплів, і 248 малих власних
напрямків були чистим шумом оцінки, який ми сумлінно клали в генератор.
⟹ обрізання рангу і здешевлює, і ВИКИДАЄ головне джерело похибки.

СХЕМА: пілот P повних прогонів → (m̂, топ-r мод) → породжуємо синтетичні h^ℓ →
проганяємо лише шари ℓ+1..31 (ціна (32−ℓ)/32).
ХВІСТ: відкинуті моди повертаємо ІЗОТРОПНО (зберігає слід) — варіант 'iso' проти 'cut'.
БЮДЖЕТ: 6485 повних прогонів; N = (6485 − P)·32/(32−ℓ).
"""
import torch, math, numpy as np, time
dev='cuda:4'; D=256; L=32; T0=time.time(); M=100000; CH=20000; BUD=6485.0
def log(m): print('[{:5.1f}s] {}'.format(time.time()-T0,m),flush=True)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d'%(n,i)].astype(np.float32),device=dev) for i in range(L)]
def pre_at(W,x,l):
    a=x
    for i in range(l): a=torch.relu(a@W[i])
    return a@W[l]
def from_pre(W,h,l):
    a=torch.relu(h)
    for i in range(l+1,L): a=torch.relu(a@W[i])
    return a

for net in (0,1):
    W=load(net); log('='*94); log('МЕРЕЖА %d'%net)
    S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(20):
        g=torch.Generator(device=dev).manual_seed(990000+c)
        z=torch.randn(50000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        S+=from_pre(W,pre_at(W,xx,0),0).double().sum(0); n+=xx.shape[0]
    TH=S/n
    log('  {:>4s} {:>6s} {:>5s} {:>5s} {:>11s} {:>11s} {:>9s} {:>11s}'
        .format('шар','пілот','ранг','хвіст','зсув MSE','дисперсія','N','SCORE'))
    for l in (24,28,30,31):
        for P in (1000,4000):
            # ПІЛОТ: антитетичні пари, повні прогони
            hs=[]
            done=0
            while done<P:
                b=min(CH,P-done)
                g=torch.Generator(device=dev).manual_seed(70000+l*7+done)
                z=torch.randn(b//2,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
                hs.append(pre_at(W,xx,l).double()); done+=b
            H=torch.cat(hs); mh=H.mean(0); Hc=H-mh
            K=(Hc.t()@Hc)/H.shape[0]
            ev,V=torch.linalg.eigh(K); ev=ev.flip(0).clamp(min=0); V=V.flip(1)
            trK=float(ev.sum())
            for r in (2,4,8,16,32,256):
                for tail in ('cut','iso'):
                    if r==256 and tail=='iso': continue
                    lam=ev[:r]; U=V[:,:r]
                    resid=max(trK-float(lam.sum()),0.0)/D if tail=='iso' else 0.0
                    Sm=torch.zeros(D,dtype=torch.float64,device=dev)
                    Q=torch.zeros(D,dtype=torch.float64,device=dev); m=0
                    for c in range(M//CH):
                        g=torch.Generator(device=dev).manual_seed(31000+c)
                        u=torch.randn(CH//2,r,device=dev,generator=g,dtype=torch.float64)
                        uu=torch.cat([u,-u],0)
                        h=mh.unsqueeze(0)+(uu*lam.sqrt().unsqueeze(0))@U.t()
                        if resid>0:
                            e=torch.randn(CH//2,D,device=dev,generator=g,dtype=torch.float64)
                            h=h+torch.cat([e,-e],0)*math.sqrt(resid)
                        a=from_pre(W,h.float(),l).double()
                        Sm+=a.sum(0); Q+=(a*a).sum(0); m+=a.shape[0]
                    est=Sm/m; bias=float(((est-TH)**2).mean())
                    vper=float((Q/m-est*est).mean())
                    N=(BUD-P)*32.0/(32.0-l)
                    mse=bias+vper/max(N,1.0)
                    log('  {:4d} {:6d} {:5d} {:>5s} {:11.3e} {:11.3e} {:9.0f} {:11.3e}'
                        .format(l,P,r,tail,bias,vper/max(N,1.0),N,0.10*mse))
    log('  орієнтир: наш score 3.09e-7; лідер 2.77e-8; 7-е місце 9.02e-8')
