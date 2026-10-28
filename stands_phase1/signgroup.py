"""ПИТАННЯ ЮЗЕРА: «а якщо кидати чисто від'ємні семпли? чи покаже це шви?»

ЧАСТИНА А (буквально): чи особливий від'ємний ортант?
   шви визначаються рядками W₀, а не осями координат -> h¹=xW₀ усе одно мішаного знаку
   прогноз: ортант нічим не особливий, а оцінка по ньому ЗМІЩЕНА

ЧАСТИНА Б (глибше): гаусіана інваріантна до перевороту знаку БУДЬ-ЯКОЇ підмножини координат
   — група 2^256. Антитетика юзає РІВНО ОДИН елемент: S=−I. Чому не більше?
   Мій аргумент: переворот k координат дає точку, тим ближчу до x, чим менше k
   -> a^L(Sx) майже = a^L(x) -> усереднення двох майже однакових ГІРШЕ за два незалежні.
   Тільки S=−I дає найдальшу точку і від'ємну кореляцію.

   ефективність пари = Var[(f(x)+f(Sx))/2] / (Var/2) = 1 + corr
      corr=−1 -> 0 (ідеал);  corr=0 -> 1 (як незалежні);  corr=+1 -> 2 (вдвічі гірше)
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/signgroup.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; DT=torch.float64
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def fwd(W,x,gates=False):
    a=x; op=[]
    for li in range(L):
        h=a@W[li]
        if gates: op.append(float((h>0).double().mean()))
        a=relu(h)
    return (a,op) if gates else a

for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    log(''); log('######## НЕТ {} ########'.format(net))

    # ---- А) від'ємний ортант ----
    N=400_000; CH=50_000
    sF=torch.zeros(D,dtype=DT,device=dev); sN=torch.zeros(D,dtype=DT,device=dev); n=0
    opF=None; opN=None
    for c in range(N//CH):
        g=torch.Generator(device=dev).manual_seed(6100+net*13+c)
        x=torch.randn(CH,D,device=dev,generator=g)
        xn=-x.abs()                                  # рівно від'ємний ортант (|N| має ту саму форму)
        aF,oF=fwd(W,x,True); aN,oN=fwd(W,xn,True)
        sF+=aF.double().sum(0); sN+=aN.double().sum(0); n+=x.shape[0]
        if opF is None: opF,opN=oF,oN
    EF=sF/n; EN=sN/n
    log('  А) ВІД\'ЄМНИЙ ОРТАНТ:')
    log('     ||E[a|x<0] − E[a]|| / ||E[a]|| = {:.4f}   (0 = ортант нічим не особливий)'.format(
        float((EN-EF).norm()/EF.norm())))
    log('     частка відкритих затворів  шар0/шар15/шар31:')
    log('        повний простір : {:.4f} {:.4f} {:.4f}'.format(opF[0],opF[15],opF[31]))
    log('        від\'ємний ортант: {:.4f} {:.4f} {:.4f}'.format(opN[0],opN[15],opN[31]))

    # ---- Б) скільки координат перевертати ----
    log('  Б) ПЕРЕВОРОТ k КООРДИНАТ (corr між a^L(x) і a^L(Sx), ефективність = 1+corr):')
    log('       k     corr     ефективність пари   читання')
    NB=200_000; CB=25_000
    for k in (1,4,16,64,128,192,255,256):
        sx=torch.zeros(D,dtype=DT,device=dev); sy=torch.zeros(D,dtype=DT,device=dev)
        sxx=torch.zeros(D,dtype=DT,device=dev); syy=torch.zeros(D,dtype=DT,device=dev)
        sxy=torch.zeros(D,dtype=DT,device=dev); m=0
        for c in range(NB//CB):
            g=torch.Generator(device=dev).manual_seed(6200+net*17+c)
            x=torch.randn(CB,D,device=dev,generator=g)
            S=torch.ones(D,device=dev); S[:k]=-1.0          # перевернути перші k координат
            a1=fwd(W,x).double(); a2=fwd(W,x*S[None,:]).double()
            sx+=a1.sum(0); sy+=a2.sum(0); sxx+=(a1*a1).sum(0)
            syy+=(a2*a2).sum(0); sxy+=(a1*a2).sum(0); m+=x.shape[0]
        mx=sx/m; my=sy/m
        vx=(sxx/m-mx*mx).clamp(min=0); vy=(syy/m-my*my).clamp(min=0)
        cv=sxy/m-mx*my
        al=vx>1e-9*float(vx.max())
        cr=float((cv[al]/(vx[al]*vy[al]).clamp(min=1e-300).sqrt()).mean())
        note='' if k not in (1,256) else ('<- майже той самий семпл' if k==1 else '<- АНТИТЕТИКА')
        log('     {:4d}   {:+.4f}      {:.4f}          {}'.format(k,cr,1+cr,note))
log('done.')
