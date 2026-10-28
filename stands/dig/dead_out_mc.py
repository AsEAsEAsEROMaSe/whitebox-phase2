"""Виправлено: вихід = ReLU(p15). Понейронний гібрид ланцюг/МК: для нейрона j МК-шум Var(out_j)/N, похибка ланцюга e_j^2.
(1) оракул Вінера по нейронах (стеля); (2) реальне правило: МК для нейронів з t15 < -TH (t з ланцюга ~ банку), інакше ланцюг;
(3) реальна Вінер-вага з оцінкою Var(out_j) за гаусовою формулою і сталою оцінкою похибки ланцюга e2c (підбір з LOO по мережах).
Справжні семпли (fp64, ReLU на всіх 16 шарах). Контроль: МК для ВИПАДКОВИХ нейронів тієї ж кількості."""
import numpy as np, torch, math
dev='cuda'; torch.set_default_dtype(torch.float64)
from math import erf
Phi=lambda x: 0.5*(1+torch.erf(x/math.sqrt(2)))
R=[]
for i in range(8):
    W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],device=dev).double()
    kp=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)
    mu=torch.tensor(kp['pre_mean'][15].astype(float),device=dev); s=torch.tensor(np.sqrt(kp['pre_m2'][15]-kp['pre_mean'][15]**2).astype(float),device=dev); t=mu/s
    ph=torch.exp(-t*t/2)/math.sqrt(2*math.pi); P=Phi(t); m1=s*(ph+t*P); varg=s*s*((1+t*t)*P+t*ph)-m1*m1   # гаусова Var(out_j)
    vtrue=torch.tensor((np.diagonal(kp['M11'][15])-kp['gt_mean'][15]**2).astype(float),device=dev)
    z=np.load('/home/trim/wb2data/pred_%d.npz'%i); pred=torch.tensor(z['pred'][15].astype(float),device=dev); gt=torch.tensor(z['gt'][15].astype(float),device=dev)
    e2=(pred-gt)**2
    g=torch.Generator(device=dev).manual_seed(77+i); a=torch.randn(1024,1024,device=dev,generator=g)
    for l in range(16): a=torch.relu(a@W[l])
    R.append(dict(e2=e2,vtrue=vtrue,varg=varg,t=t,a=a,pred=pred,gt=gt))
    print('дамп %d: corr(log varg, log vtrue) %.3f | частка MSE у нейронах t<-3: %.3f (їх %d)'%(i,np.corrcoef(np.log(varg.cpu().numpy()+1e-30),np.log(vtrue.clamp_min(1e-30).cpu().numpy()))[0,1],float(e2[t<-3].sum()/e2.sum()),int((t<-3).sum())))
for N in [64,256,1024]:
    orc=hard={}
    o=[];hb={2.0:[],3.0:[],4.0:[]};cb=[]
    for d in R:
        base=float(d['e2'].mean()); noise=d['vtrue']/N
        o.append((base,float((d['e2']*noise/(d['e2']+noise)).mean())))
        mc=d['a'][:N].mean(0)
        for TH in hb:
            m=d['t']<-TH; new=torch.where(m,mc,d['pred']); hb[TH].append((base,float(((new-d['gt'])**2).mean())))
        m=d['t']<-3.0; perm=torch.randperm(1024,device=dev)[:int(m.sum())]; r=torch.zeros(1024,dtype=torch.bool,device=dev); r[perm]=True
        cb.append((base,float(((torch.where(r,mc,d['pred'])-d['gt'])**2).mean())))
    f=lambda v: sum(x for x,_ in v)/sum(y for _,y in v)
    print('N=%4d (%.2f%%B): оракул Вінера x%.3f | МК для t<-2: x%.3f · t<-3: x%.3f · t<-4: x%.3f | контроль випадкові нейрони x%.4f'%(N,100*N*3.36e7/2.199e12,f(o),f(hb[2.0]),f(hb[3.0]),f(hb[4.0]),f(cb)))
