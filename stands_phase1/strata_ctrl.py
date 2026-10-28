"""ПОЗИТИВНИЙ КОНТРОЛЬ до strata.py: чи БАЧИТЬ колонка «неквадратичний 1D» відомий 4-й порядок?
Підсаджуємо r' = r + c·He4(t) вздовж активного напрямку. Якщо колонка не спалахне —
прилад зламаний, і нуль у strata.py нічого не означає."""
import torch,math,numpy as np,time
exec(open('/mnt/win_d/llm/White-Box/strata.py').read().split('for net in (0,1):')[0])
W=load(0)
M=active_subspace(W,20000); ev,V=torch.linalg.eigh(M); V=V.flip(1)
U=V[:,:1].float()
CH=20000
t_all=[];r_all=[]
for c in range(NMEAS//CH):
    g=torch.Generator(device=dev).manual_seed(52000+c)
    x=torch.randn(CH,D,device=dev,generator=g)
    with torch.no_grad(): r=even_part(W,x).double()
    t_all.append(x.double()@U.double()); r_all.append(r)
T=torch.cat(t_all,0); R=torch.cat(r_all,0); R=R-R.mean(0,keepdim=True)
t=T[:,0]
sd=float(R.std(0).mean())
He4=t**4-6*t**2+3
He2=t**2-1
for name,extra in (('чистий r',0*t),('r + 0.05σ·He4',0.05*sd*He4/math.sqrt(24)),
                   ('r + 0.05σ·He2 (має піти в квадр.)',0.05*sd*He2/math.sqrt(2))):
    R2=R+extra.unsqueeze(1)
    R2=R2-R2.mean(0,keepdim=True)
    vtot=float((R2*R2).mean(0).sum())
    idx=torch.argsort(t); lab=torch.empty_like(idx); lab[idx]=torch.arange(t.shape[0],device=dev)
    lab=(lab*NB)//t.shape[0]
    oh=torch.zeros(NB,t.shape[0],dtype=torch.float64,device=dev)
    oh[lab,torch.arange(t.shape[0],device=dev)]=1.0
    cnt=oh.sum(1,keepdim=True); mb=(oh@R2)/cnt
    raw=float(((mb*mb)*cnt).sum()/t.shape[0])
    tb=(oh@t.unsqueeze(1))/cnt; h=(tb*tb-1.0).squeeze(1)
    w=cnt.squeeze(1)/t.shape[0]; h=h-(w*h).sum(); nrm=float((w*h*h).sum())
    proj=((w.unsqueeze(1)*h.unsqueeze(1)*mb).sum(0))/max(nrm,1e-30)
    res=mb-h.unsqueeze(1)*proj.unsqueeze(0); res=res-(w.unsqueeze(1)*res).sum(0,keepdim=True)
    nonq=float(((res*res)*cnt).sum()/t.shape[0])
    log('  {:38s} 1D {:7.3f}%   неквадр {:7.3f}%'.format(name,100*raw/vtot,100*nonq/vtot))
