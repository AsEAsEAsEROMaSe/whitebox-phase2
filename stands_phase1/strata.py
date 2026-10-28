"""СТРАТИФІКАЦІЯ (інший клас, не поліноміальний): чи має ПАРНИЙ залишок жирні напрямки?

ЛОГІКА. v2 = антитетика (вбиває непарну частину побітово) + ZCA (вбиває квадратичну ТОЧНО).
Лишається 43-49% у 4-му порядку й вище, і [[finding-fourth-order-has-no-cheap-handle]]
каже: дешевого ПОЛІНОМІАЛЬНОГО важеля на нього нема.

Стратифікація — інструмент з ІНШОГО берега: не міняємо оцінювач, міняємо МІРУ.
Якщо розбити вісь t=x·u на квантильні шари й брати рівно по семплу з шару, то вздовж u
відтворюється ВСЯ одновимірна маргіналь — усі порядки, не лише 2-й.
  • незміщено для БУДЬ-ЯКОГО u (немає якоря, який треба купувати ⟹ [[finding-anchor-cost-identity]] не діє)
  • поганий u = менший виграш, ніколи не штраф
  • форма = той самий великий матмул

ЩО МІРЯЄМО. Виграш від ідеальної 1D-стратифікації вздовж u = частка дисперсії, яку
пояснює УМОВНЕ СЕРЕДНЄ m(t)=E[r|x·u=t], де r — ПАРНА частина (те, що бачить v2).
З неї треба відняти квадратичний шматок (його ZCA вже забирає) ⟹ звітуємо
«неквадратичний 1D-вміст» — це і є ЧИСТИЙ приріст понад v2.

КАНДИДАТИ u:
  A) активний підпростір: топ-власні вектори M=E[∇r ∇rᵀ], оцінені випадковими пробами
     v~N(0,I):  ∇(r·v) ⟹ E_v[∇(r·v)∇(r·v)ᵀ] = JᵀJ.  Одна проба на семпл.
  B) випадкові напрямки — КОНТРОЛЬ (скільки дає «нізвідки»)
  C) shuffle-контроль на мітках шарів — бо сира частка завищена дрібністю шарів
     ([[feedback-isolated-bench-lies]], правило С47)
"""
import torch, time, math, numpy as np
LOG='/mnt/win_d/llm/White-Box/strata.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(W,x):
    a=x
    for i in range(L): a=torch.relu(a@W[i])
    return a

NB=40            # квантильних шарів
NPROBE=20000     # семплів на оцінку активного підпростору
NMEAS=200000     # семплів на вимір частки

def even_part(W,x):
    """r(x) = ½(y(x)+y(−x)) — рівно те, що бачить антитетичний оцінювач"""
    return 0.5*(fwd(W,x)+fwd(W,-x))

def active_subspace(W,n):
    """M = Σ ∇(r·v)∇(r·v)ᵀ через випадкові проби v"""
    M=torch.zeros(D,D,dtype=torch.float64,device=dev)
    CH=2000
    for c in range(n//CH):
        g=torch.Generator(device=dev).manual_seed(31000+c)
        x=torch.randn(CH,D,device=dev,generator=g,requires_grad=True)
        v=torch.randn(CH,D,device=dev,generator=g)
        r=even_part(W,x)
        s=(r*v).sum()
        gr,=torch.autograd.grad(s,x)
        gr=gr.double()
        M+=gr.t()@gr
    return M/n

def content(W,U,tag):
    """для кожного напрямку u: частка Var(r), яку бере умовне середнє по шарах,
       і та сама частка ПІСЛЯ зняття {1, He2} (те, що ZCA вже має)"""
    K=U.shape[1]
    t_all=[]; r_all=[]
    CH=20000
    for c in range(NMEAS//CH):
        g=torch.Generator(device=dev).manual_seed(52000+c)
        x=torch.randn(CH,D,device=dev,generator=g)
        with torch.no_grad():
            r=even_part(W,x).double()
        t_all.append((x.double()@U.double())); r_all.append(r)
    T=torch.cat(t_all,0); R=torch.cat(r_all,0)
    R=R-R.mean(0,keepdim=True)
    vtot=float((R*R).mean(0).sum())
    out=[]
    for k in range(K):
        t=T[:,k]
        # квантильні шари
        idx=torch.argsort(t)
        lab=torch.empty_like(idx); lab[idx]=torch.arange(t.shape[0],device=dev)
        lab=(lab*NB)//t.shape[0]
        onehot=torch.zeros(NB,t.shape[0],dtype=torch.float64,device=dev)
        onehot[lab,torch.arange(t.shape[0],device=dev)]=1.0
        cnt=onehot.sum(1,keepdim=True)
        mb=(onehot@R)/cnt                                  # NB×256 середні по шарах
        raw=float(((mb*mb)*cnt).sum()/t.shape[0])
        # shuffle-контроль: та сама статистика на перемішаних мітках
        gsh=torch.Generator(device=dev).manual_seed(9000+k)
        perm=torch.randperm(t.shape[0],device=dev,generator=gsh)
        oh2=onehot[:,perm]
        mb2=(oh2@R)/cnt
        sh=float(((mb2*mb2)*cnt).sum()/t.shape[0])
        # зняти {1, He2(t)} — квадратичну частину, яку ZCA вже бере
        tb=(onehot@t.double().unsqueeze(1))/cnt            # середнє t у шарі
        he2=tb*tb-1.0
        w=cnt.squeeze(1)/t.shape[0]
        h=he2.squeeze(1); h=h-(w*h).sum()
        nrm=float((w*h*h).sum())
        proj=((w.unsqueeze(1)*h.unsqueeze(1)*mb).sum(0))/max(nrm,1e-30)
        res=mb-h.unsqueeze(1)*proj.unsqueeze(0)
        res=res-(w.unsqueeze(1)*res).sum(0,keepdim=True)
        nonq=float(((res*res)*cnt).sum()/t.shape[0])
        out.append((raw-sh,nonq-sh,sh))
    log('  {:s}'.format(tag))
    log('    {:>4s} {:>14s} {:>16s} {:>12s}'.format('u','1D вміст','неквадр. 1D','shuffle'))
    for k,(a,b,s) in enumerate(out):
        log('    {:4d} {:13.3f}% {:15.3f}% {:11.3f}%'
            .format(k,100*a/vtot,100*b/vtot,100*s/vtot))
    tot=sum(max(b,0) for _,b,_ in out)
    log('    СУМА неквадратичного вмісту по {:d} напрямках: {:.2f}% від Var(r)'.format(K,100*tot/vtot))
    return tot/vtot

for net in (0,1):
    W=load(net); log('='*80); log('МЕРЕЖА %d'%net)
    ta=time.time()
    M=active_subspace(W,NPROBE)
    ev,V=torch.linalg.eigh(M)
    ev=ev.flip(0); V=V.flip(1)
    log('  активний підпростір за {:.1f}с; спектр (топ-8 / хвіст):'.format(time.time()-ta))
    log('    {:s}'.format(' '.join('{:.3e}'.format(float(e)) for e in ev[:8])))
    log('    частка топ-8 у сліді: {:.1f}%   топ-32: {:.1f}%'
        .format(100*float(ev[:8].sum()/ev.sum()),100*float(ev[:32].sum()/ev.sum())))
    K=6
    Ua=V[:,:K].float()
    g=torch.Generator(device=dev).manual_seed(777)
    Ur,_=torch.linalg.qr(torch.randn(D,K,device=dev,generator=g))
    a=content(W,Ua,'A) активний підпростір (топ-%d)'%K)
    b=content(W,Ur,'B) ВИПАДКОВІ напрямки — КОНТРОЛЬ')
    log('  ⟹ приріст понад v2, якщо стратифікувати {:d} напрямків: {:.2f}% (контроль {:.2f}%)'
        .format(K,100*a,100*b))
log('done.')
